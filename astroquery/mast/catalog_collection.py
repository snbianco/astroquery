from pyvo.dal import TAPService, DALQueryError
from astropy.table import Table
from astroquery import log
import difflib

from astroquery.exceptions import InvalidQueryError

__all__ = ['CatalogCollection']

class CatalogCollection:

    _TAP_BASE_URL = 'https://masttest.stsci.edu/vo-tap/api/v0.1/'

    def __init__(self, collection):
        """Initialize the CatalogCollection with a list of catalogs.

        Parameters
        ----------
        collection : str
            A list of catalog objects retrieved from MAST.
        """
        self.name = collection.strip().lower()
        self.tap_service = TAPService(self._TAP_BASE_URL + self.name)
        self._catalog_metadata_cache = dict()

        self.catalogs = self._get_catalogs()
        self.catalog_names = self.catalogs['catalog_name'].tolist()
        self.supported_adql_functions = self._get_adql_supported_functions()
        self.default_catalog = self.get_default_catalog()

    def _get_catalogs(self):
        """
        Retrieve the list of catalogs in this collection.

        Returns
        -------
        list of str
            List of catalog names.
        """        
        log.debug(f"Fetching available tables for collection '{self.name}' from MAST TAP service.")
        tables = self.tap_service.tables
        names = [t.name for t in tables]
        descriptions = [t.description for t in tables]
        
        # Create an Astropy Table to hold the results
        result_table = Table([names, descriptions], names=('catalog_name', 'description'))
        return result_table
    
    def _get_column_metadata(self, catalog):
        """
        For a given catalog, return metadata about the table.

        Parameters
        ----------
        catalog : str
            The catalog within the collection to get metadata for.

        Returns
        -------
        response : `~astropy.table.Table`
            A table containing metadata about the specified table, including column names, data types, and descriptions.
        """
        log.debug(f"Fetching column metadata for collection '{self.name}', catalog '{catalog}' from MAST TAP service.")
        tap_table = next((t for name, t in self.tap_service.tables.items() if name.lower() == catalog.lower()), None)

        # Extract column metadata
        col_names = [col.name for col in tap_table.columns]
        # Some pyvo versions store datatype differently; fall back gracefully
        col_datatypes = []
        for col in tap_table.columns:
            try:
                col_datatypes.append(col.datatype._content)
            except AttributeError:
                # Fallback: str(col.datatype) or None
                col_datatypes.append(getattr(col.datatype, '_content', str(col.datatype)))
        col_units = [col.unit for col in tap_table.columns]
        col_ucds = [col.ucd for col in tap_table.columns]
        col_descriptions = [col.description for col in tap_table.columns]

        # Create an Astropy Table to hold the metadata
        column_metadata = Table([col_names, col_datatypes, col_units, col_ucds, col_descriptions],
                               names=('name', 'data_type', 'unit', 'ucd', 'description'))
        return column_metadata
    
    def _get_ra_dec_column_names(self, column_metadata):
        """
        Return the RA and Dec column names for a given catalog and table.

        Parameters
        ----------
        catalog : str
            The catalog to be queried.
        table : str
            The table within the catalog to query.

        Returns
        -------
        tuple
            A tuple containing the (ra_column, dec_column) names.
        """
        # Look for a column with UCD 'pos.eq.ra' and 'pos.eq.dec'
        ra_col = None
        dec_col = None
        for name, ucd in zip(column_metadata['name'], column_metadata['ucd']):
            if ucd and 'pos.eq.ra;meta.main' in ucd:
                #TODO: ps1_dr2.mean_object and ps1_dr2.stacked_object has a column that can be used, but is not labeled with "meta.main"
                ra_col = name
            elif ucd and 'pos.eq.dec;meta.main' in ucd:
                dec_col = name
        return ra_col, dec_col
    
    def _get_adql_supported_functions(self):
        """
        Retrieve the ADQL supported functions of the TAP service.

        Returns
        -------
        list
            A list of supported ADQL functions.
        """
        adql_functions = ['CIRCLE', 'POLYGON', 'POINT', 'CONTAINS', 'INTERSECTS']
        supported = []
        capabilities = self.tap_service.capabilities
        for cap in capabilities:
            if cap.standardid == 'ivo://ivoa.net/std/TAP':  # TAP is supported
                for lang in cap.languages:
                    if lang.name == 'ADQL':  # ADQL is supported
                        for func in adql_functions:
                            if lang.get_feature('ivo://ivoa.net/std/TAPRegExt#features-adqlgeo', func):
                                supported.append(func)
        return supported       

    def get_catalog_metadata(self, catalog):
        if catalog in self._catalog_metadata_cache:
            return self._catalog_metadata_cache[catalog]
        
        self._verify_catalog(catalog)
        
        metadata = self._get_column_metadata(catalog)
        ra_col, dec_col = self._get_ra_dec_column_names(metadata)

        supports_spatial_queries = True
        if ra_col is None or dec_col is None:
            supports_spatial_queries = False
        else:
            # Test spatial query support
            spatial_query = (f'SELECT TOP 0 * FROM {catalog} WHERE CONTAINS(POINT(\'ICRS\', {ra_col}, {dec_col}), '
                             'CIRCLE(\'ICRS\', 0, 0, 0.1)) = 1')
            try:
                self.tap_service.search(spatial_query)
            except DALQueryError:
                supports_spatial_queries = False

        # TODO: Get principal columns
        self._catalog_metadata_cache[catalog] = {
            'column_metadata': metadata,
            'ra_column': ra_col,
            'dec_column': dec_col,
            'supports_spatial_queries': supports_spatial_queries
        }
        return self._catalog_metadata_cache[catalog]
    
    def _verify_catalog(self, catalog):
        """
        Verify that the specified catalog is valid for the given collection.

        Parameters
        ----------
        collection : CatalogCollection
            The collection to be verified.
        catalog : str
            The catalog to be verified.

        Raises
        ------
        InvalidQueryError
            If the specified catalog is not valid for the given collection.
        """
        lower_map = {name.lower(): name for name in self.catalog_names}
        if catalog.lower() not in lower_map:
            closest_match = difflib.get_close_matches(catalog, self.catalog_names, n=1)
            error_msg = (
                f"Catalog '{catalog}' is not recognized for collection '{self.name}'. Did you mean '{closest_match[0]}'?"
                if closest_match
                else f"Catalog '{catalog}' is not recognized for collection '{self.name}'."
            )
            error_msg += " Available catalogs are: " + ", ".join(self.catalog_names)
            raise InvalidQueryError(error_msg)
        
    def _verify_criteria(self, catalog, **criteria):
        """
        Check that criteria keyword arguments are valid column names for the specified collection and catalog.

        Parameters
        ----------
        catalog : str
            The catalog within the collection to query.
        **criteria
            Keyword arguments representing criteria filters to apply.

        Raises
        ------
        InvalidQueryError
            If a keyword does not match any valid column names, an error is raised that suggests the closest
            matching column name, if available.
        """
        if not criteria:
            return
        self._verify_catalog(catalog)
        col_names = list(self.get_catalog_metadata(catalog)['column_metadata']['name'])

        # Check each criteria argument for validity
        for kwd in criteria.keys():
            if kwd not in col_names:
                closest_match = difflib.get_close_matches(kwd, col_names, n=1)
                error_msg = (
                    f"Filter '{kwd}' is not recognized for collection '{self.name}' and catalog '{catalog}'. Did you mean '{closest_match[0]}'?"
                    if closest_match
                    else f"Filter '{kwd}' is not recognized for collection '{self.name}' and catalog '{catalog}'."
                )
                raise InvalidQueryError(error_msg)

    def get_default_catalog(self):
        # Pick default catalog = first one that does NOT start with "tap_schema."
        default_catalog = next((c for c in self.catalog_names if not c.startswith("tap_schema.")), None)

        # If no valid catalog found, fallback to the first one
        if default_catalog is None:
            default_catalog = self.catalog_names[0] if self.catalog_names else None

        return default_catalog
    
    def run_tap_query(self, adql):
        """
        Run a TAP query against the specified catalog.

        Parameters
        ----------
        adql : str
            The ADQL query string.

        Returns
        -------
        response : `~astropy.table.Table`
            The result of the TAP query as an Astropy Table.
        """
        log.debug(f"Running TAP query on collection '{self.name}': {adql}")
        result = self.tap_service.search(adql)
        return result.to_table()
