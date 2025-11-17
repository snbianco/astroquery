# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
MAST Collections
================

This module contains various methods for querying MAST collections such as catalogs.
"""

import difflib
import warnings
import re
from collections.abc import Iterable

import astropy.units as u
import astropy.coordinates as coord
import requests
from astropy.table import Table
from regions import CircleSkyRegion, PolygonSkyRegion

from astroquery import log
from ..utils import async_to_sync
from ..exceptions import InputWarning, InvalidQueryError, MaxResultsWarning

from . import utils
from .catalog_collection import CatalogCollection
from .core import MastQueryWithLogin


__all__ = ['Catalogs', 'CatalogsClass']


@async_to_sync
class CatalogsClass(MastQueryWithLogin):
    """
    MAST catalog query class.

    Class for querying MAST catalog data.
    """

    def __init__(self, collection="hsc", catalog=None):

        super().__init__()

        services = {"panstarrs": {"path": "panstarrs/{data_release}/{table}.json",
                                  "args": {"data_release": "dr2", "table": "mean"}}}
        self._catalogs_mast_search_options = ['columns', 'sort_by', 'table', 'data_release']

        self._service_api_connection.set_service_params(services, "catalogs", True)

        self.catalog_limit = None
        self._current_connection = None
        self._service_columns = dict()  # Info about columns for Catalogs.MAST services


        self._no_longer_supported_collections = ['ctl', 'diskdetective', 'galex', 'plato']
        self.available_collections = self.get_collections()['collection_name'].tolist()
        self._collections_cache = dict()

        self._collection = None
        self._catalog = None
        self.collection = collection
        if catalog:
            self.catalog = catalog

    @property
    def collection(self):
        return self._collection

    @collection.setter
    def collection(self, collection):
        collection_obj = self._get_collection_obj(collection)
        self._collection = collection_obj

        # Only change catalog if not set yet or invalid for this collection
        if not hasattr(self, "_catalog") or self._catalog not in collection_obj.catalog_names:
            self._catalog = collection_obj.default_catalog

    @property    
    def catalog(self):
        return self._catalog
    
    @catalog.setter
    def catalog(self, catalog):
        # Setter that updates the service parameters if the catalog is changed
        self.collection._verify_catalog(catalog)
        log.debug(f"Set catalog to: {catalog}")
        self._catalog = catalog

    def get_collections(self):
        """
        Return a list of available collections from MAST.

        Returns
        -------
        response : list of str
            A list of available MAST collections.
        """
        # If already cached, use it directly
        if getattr(self, "available_collections", None):
            return Table([self.available_collections], names=('collection_name',))

        # Otherwise, fetch from the TAP service
        log.debug("Fetching available collections from MAST TAP service.")
        url = "https://masttest.stsci.edu/vo-tap/api/v0.1/openapi.json"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        # Extract collection enumeration
        collection_enum = data["components"]["schemas"]["CatalogName"]["enum"]

        # Cache the results
        self.available_collections = collection_enum

        # Build an Astropy Table to hold the results
        collection_table = Table([collection_enum], names=('collection_name',))
        return collection_table

    def get_catalogs(self, collection=None):
        """
        For a given Catalogs.MAST collection, return a list of available catalogs.

        Parameters
        ----------
        collection : str
            The collection to be queried.

        Returns
        -------
        response : list of str
            A list of available catalogs for the specified collection.
        """
        # If no collection specified, use the class attribute
        collection_obj = self._get_collection_obj(collection) if collection else self.collection
        return collection_obj.catalogs
    
    def get_catalog_metadata(self, collection=None, catalog=None):
        """
        For a given Catalogs.MAST collection and catalog, return metadata about the catalog.

        Parameters
        ----------
        collection : str
            The collection to be queried.
        catalog : str
            The catalog within the collection to get metadata for.

        Returns
        -------
        response : `~astropy.table.Table`
            A table containing metadata about the specified catalog, including column names, data types, and descriptions.
        """
        collection_obj, catalog = self._parse_inputs(collection, catalog)
        return collection_obj.get_catalog_metadata(catalog)['column_metadata']
    
    def query_criteria_tap(self, collection=None, *, catalog=None, coordinates=None, region=None, objectname=None, 
                           radius=0.2*u.deg, resolver=None, limit=5000, offset=0, count_only=False, select_cols=None, **criteria):
        """
        Query a MAST catalog using criteria filters via TAP.

        Parameters
        ----------
        collection : str
            The collection to be queried.
        catalog : str
            The catalog within the collection to query.
        coordinates : str or `~astropy.coordinates` object, optional
            The target around which to search. It may be specified as a string (e.g., '350 -80') or as an Astropy coordinates object.
        region : str | iterable | `~astropy.regions.Region`, optional
            The region to search within. It may be specified as a string (e.g., 'circle(350 -80, 0.2d)') or as an Astropy regions object.
        objectname : str, optional
            The name of the object to resolve and search around.
        radius : str or `~astropy.units.Quantity` object, optional
            The search radius around the target coordinates or object. Default 0.2 degrees.
        resolver : str, optional
            The name resolver service to use when resolving ``objectname``.
        limit : int
            The maximum number of results to return. Default is 5000.
        offset : int
            The number of rows to skip before starting to return rows. Default is 0.
        count_only : bool
            If True, only return the count of matching records instead of the records themselves. Default is False.
        select_cols : list of str, optional
            List of column names to include in the result. If None or empty, all columns are returned.
        **criteria
            Keyword arguments representing criteria filters to apply.

                Criteria syntax
                ----------------
                - Strings support wildcards using '*' (converted to SQL '%') and '%'.
                - Lists are combined with OR for positive values; empty lists yield no matches.
                - Numeric columns support comparison operators ('<', '<=', '>', '>=') and inclusive ranges using
                    the syntax 'low..high' (e.g., '5..10'). Mixed lists of numbers and comparisons are OR-combined.
                - Negation: Prefix any value with '!' to negate that predicate. For list inputs, all negated values
                    for the same column are AND-combined, then ANDed with the OR of the positive values:
                        (neg1 AND neg2 AND ...) AND (pos1 OR pos2 OR ...).

                Examples
                --------
                - file_suffix=['A', 'B', '!C'] -> (file_suffix != 'C') AND (file_suffix IN ('A', 'B'))
                - size=['!14400', '<20000'] -> (size != 14400) AND (size < 20000)

        Returns
        -------
        response : `~astropy.table.Table`
            A table containing the query results.
        """
        # Should not specify both region and coordinates
        if coordinates and region:
            raise InvalidQueryError('Specify either `region` or `coordinates`, not both.')
        
        # Should not specify both region and objectname
        if objectname and region:
            raise InvalidQueryError('Specify either `region` or `objectname`, not both.')
        
        collection_obj, catalog = self._parse_inputs(collection, catalog)
        collection_obj._verify_criteria(catalog, **criteria)

        columns = '*' if not select_cols else self._parse_select_cols(select_cols, collection_obj.get_catalog_metadata(catalog)['column_metadata'])
        adql = f'SELECT TOP {limit} {columns} FROM {catalog.lower()} ' if not count_only else f'SELECT COUNT(*) AS count_all FROM {catalog.lower()} '
        if region or coordinates or objectname:
            # Check if the catalog supports spatial queries
            if not collection_obj.get_catalog_metadata(catalog).get('supports_spatial_queries', False):
                raise InvalidQueryError(f"Catalog '{catalog}' in collection '{collection_obj.name}' does not support spatial queries.")

            # Positional query
            adql_region = ''
            if region:
                adql_region = self._create_adql_region(region)
            if objectname or coordinates:  # Cone search
                coordinates = utils.parse_input_location(coordinates=coordinates,
                                                         objectname=objectname,
                                                         resolver=resolver)
                radius = coord.Angle(radius, u.deg)  # If radius is just a number we assume degrees
                adql_region = f'CIRCLE(\'ICRS\', {coordinates.ra.deg}, {coordinates.dec.deg}, {radius.to(u.deg).value})'

            # Get RA/Dec column names
            ra_col = collection_obj.get_catalog_metadata(catalog)['ra_column']
            dec_col = collection_obj.get_catalog_metadata(catalog)['dec_column']
            adql += (f'WHERE CONTAINS(POINT(\'ICRS\', {ra_col}, {dec_col}), {adql_region}) = 1 ')

        # Add additional constraints
        if criteria:
            conditions = self._format_criteria_conditions(collection, catalog, criteria)
            if 'WHERE' in adql:
                adql += 'AND ' + ' AND '.join(conditions)
            else:
                adql += 'WHERE ' + ' AND '.join(conditions)

        result = collection_obj.run_tap_query(adql)

        if count_only:
            return result['count_all'][0]
        return result
    
    def query_region_tap(self, coordinates=None, *, radius=0.2*u.deg, collection=None, catalog=None, 
                         region=None, limit=5000, offset=0, count_only=False, select_cols=None, **criteria):        
        # Must specify one of region or coordinates
        if region is None and coordinates is None:
            raise InvalidQueryError('Must specify either `region` or `coordinates`.')
        
        return self.query_criteria_tap(collection=collection,
                                       catalog=catalog,
                                       coordinates=coordinates,
                                       region=region,
                                       radius=radius,
                                       limit=limit,
                                       offset=offset,
                                       count_only=count_only,
                                       select_cols=select_cols,
                                       **criteria)

    def query_object_tap(self, objectname, *, radius=0.2*u.deg, collection=None, catalog=None, resolver=None,
                         limit=5000, offset=0, count_only=False, select_cols=None, **criteria):
        return self.query_criteria_tap(collection=collection,
                                       catalog=catalog,
                                       objectname=objectname,
                                       radius=radius,
                                       resolver=resolver,
                                       limit=limit,
                                       offset=offset,
                                       count_only=count_only,
                                       select_cols=select_cols,
                                       **criteria)
    
    # def _parse_result(self, response, *, verbose=False):
    #     results_table = self._current_connection._parse_result(response, verbose=verbose)
    #     if len(results_table) == self.catalog_limit:
    #         warnings.warn("Maximum catalog results returned, may not include all sources within radius.",
    #                       MaxResultsWarning)
    #     return results_table

    def _verify_collection(self, collection):
        """
        Verify that the specified collection is valid.

        Parameters
        ----------
        collection : str
            The collection to be verified.

        Raises
        ------
        InvalidQueryError
            If the specified collection is not valid.
        """
        if collection.lower() not in self.available_collections:
            if collection in self._no_longer_supported_collections:
                error_msg = (f"Collection '{collection}' is no longer supported. To query from this catalog, "
                             f"please use a version of Astroquery older than 0.4.12.")
            else:
                closest_match = difflib.get_close_matches(collection, self.available_collections, n=1)
                error_msg = (
                    f"Collection '{collection}' is not recognized. Did you mean '{closest_match[0]}'?"
                    if closest_match
                    else f"Collection '{collection}' is not recognized."
                )
            error_msg += " Available collections are: " + ", ".join(self.available_collections)
            raise InvalidQueryError(error_msg)
        
    def _get_collection_obj(self, collection_name):
        """
        Given a collection name, find or create the corresponding CatalogCollection object.
        """
        collection_name = collection_name.lower().strip()
        if collection_name in self._collections_cache:
            log.debug("Using cached CatalogCollection for collection: " + collection_name)
            return self._collections_cache[collection_name]
        
        self._verify_collection(collection_name)
        collection_obj = CatalogCollection(collection_name)
        log.debug("Cached CatalogCollection for collection: " + collection_name)
        self._collections_cache[collection_name] = collection_obj
        return collection_obj

    def _parse_inputs(self, collection=None, catalog=None):
        """
        Return (collection, catalog) applying default attributes, validation, and normalization.

        Parameters
        ----------
        collection : str, optional
            The collection to be queried. If None, uses the instance's default collection.
        catalog : str, optional
            The catalog within the collection to query. If None, uses the instance's default catalog.

        Returns
        -------
        tuple
            A tuple containing the (collection, catalog) to be queried.
        """

        collection_obj = self._get_collection_obj(collection) if collection else self.collection

        if not catalog:
            # If the class attribute catalog is valid for the collection, use it
            # Otherwise, use the default catalog for the collection
            if self.catalog in collection_obj.catalog_names:
                catalog = self.catalog
            else:
                catalog = collection_obj.default_catalog
        else:
            catalog = catalog.lower()
            collection_obj._verify_catalog(catalog)

        return collection_obj, catalog
    
    def _parse_select_cols(self, select_cols, column_metadata):
        """
        Validate and parse the select_cols parameter.

        Parameters
        ----------
        select_cols : list of str
            List of column names to include in the result.
        catalog_metadata : `~astropy.table.Table`
            Metadata table for the catalog.

        Returns
        -------
        str
            Comma-separated string of valid column names for ADQL SELECT clause.

        Raises
        ------
        InvalidQueryError
            If any specified column is not found in the catalog metadata.
        """
        valid_columns = column_metadata['name'].tolist()
        valid_selected = []
        for col in select_cols:
            if col not in valid_columns:
                closest_match = difflib.get_close_matches(col, valid_columns, n=1)
                if closest_match:
                    warnings.warn(f"Column '{col}' not found in catalog. Did you mean '{closest_match[0]}'?", InputWarning)
                else:
                    warnings.warn(f"Column '{col}' not found in catalog.", InputWarning)
            else:
                valid_selected.append(col)
        return ', '.join(valid_selected)
    
    def _create_adql_region(self, region):
        """
        Returns the ADQL description of the given polygon or circle region.

        Parameters
        ----------
        region : str | iterable | astropy.regions.Region
            - Iterable of RA/Dec pairs as lists/sequences
            - STC-S POLYGON or CIRCLE string
            - astropy region (PolygonSkyRegion, CircleSkyRegion, etc.)

        Returns
        -------
        adql_region : str
            ADQL representation of the region (POLYGON or CIRCLE)
        """
        # Case 1: region is a string (e.g. STC-S syntax)
        if isinstance(region, str):
            region = region.strip().lower()
            parts = region.split()

            if parts[0] == 'polygon':
                # Handle POLYGON (with or without coord frame)
                try:
                    float(parts[1])  # Check if next token is numeric
                    point_parts = parts[1:]
                except ValueError:
                    point_parts = parts[2:]  # skip frame name if present
                point_string = ','.join(point_parts)
                adql_region = f"POLYGON('ICRS',{point_string})"
            elif parts[0] == 'circle':
                # Handle CIRCLE (with or without coord frame)
                try:
                    float(parts[1])
                    ra, dec, radius = parts[1], parts[2], parts[3]
                except ValueError:
                    ra, dec, radius = parts[2], parts[3], parts[4]
                adql_region = f"CIRCLE('ICRS',{ra},{dec},{radius})"
            else:
                raise ValueError(f"Unrecognized region string: {region}")

        # Case 2: region is an astropy region object
        elif isinstance(region, CircleSkyRegion):
            center = region.center.icrs
            radius = region.radius.to(u.deg).value
            adql_region = (
                f"CIRCLE('ICRS',{center.ra.deg},{center.dec.deg},{radius})"
            )
        elif isinstance(region, PolygonSkyRegion):
            verts = region.vertices.icrs
            point_string = ','.join(f"{v.ra.deg},{v.dec.deg}" for v in verts)
            adql_region = f"POLYGON('ICRS',{point_string})"

        # Case 3: region is an iterable of coordinate pairs
        elif isinstance(region, Iterable):
            # Expect something like [(ra1, dec1), (ra2, dec2), ...]
            try:
                flat_points = [float(x) for point in region for x in point]
            except Exception as e:
                raise ValueError(f"Invalid iterable region format: {region}") from e

            point_string = ','.join(str(x) for x in flat_points)
            adql_region = f"POLYGON('ICRS',{point_string})"

        else:
            raise TypeError(f"Unsupported region type: {type(region)}")

        return adql_region
    
    # ---- Formatting helpers extracted for readability ----
    def _get_numeric_columns(self, catalog, table):
        """
        Return a set of column names with numeric types for a given table.
        Relies on metadata types to detect numeric columns.
        """
        meta = self.get_catalog_metadata(catalog, table)
        num_types = (
            'int', 'integer', 'smallint', 'bigint', 'tinyint',
            'float', 'double', 'double precision', 'real', 'numeric', 'decimal'
        )
        return {
            n for n, t in zip(meta['name'], meta['data_type'])
            if isinstance(t, str) and any(nt in t.lower() for nt in num_types)
        }

    def _quote_sql_string(self, s: str) -> str:
        """Escape single quotes per SQL (double them)."""
        return s.replace("'", "''")

    def _parse_numeric_expr(self, s: str):
        """
        Parse numeric comparison/range string.
        Returns one of:
            - ('between', low, high)
            - ('cmp', op, num)
            - None if not a recognized pattern
        """
        if not isinstance(s, str):
            return None
        ss = s.strip()
        m = re.fullmatch(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*\.\.\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)", ss)
        if m:
            return ('between', m.group(1), m.group(2))
        m = re.fullmatch(r"(<=|>=|<|>)\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)", ss)
        if m:
            return ('cmp', m.group(1), m.group(2))
        return None

    def _format_scalar_predicate(self, col, val, numeric_cols):
        """Build predicate for a scalar value (handles negation and wildcards)."""
        if isinstance(val, bool):
            return f"{col} = {int(val)}"
        if isinstance(val, str):
            is_neg = val.startswith('!')
            sval = val[1:].strip() if is_neg else val
            if col in numeric_cols:
                parsed = self._parse_numeric_expr(sval)
                if parsed is not None:
                    if parsed[0] == 'between':
                        expr = f"{col} BETWEEN {parsed[1]} AND {parsed[2]}"
                    else:
                        expr = f"{col} {parsed[1]} {parsed[2]}"
                    return f"NOT ({expr})" if is_neg else expr
                try:
                    num = float(sval)
                    if sval.strip().isdigit():
                        expr = f"{col} = {int(num)}"
                    else:
                        expr = f"{col} = {num}"
                    return f"NOT ({expr})" if is_neg else expr
                except Exception:
                    raise InvalidQueryError(
                        f"Column '{col}' is numeric; unsupported value '{val}'. Use numbers, comparisons like '<10', or ranges like '5..10'."
                    )
            # Non-numeric
            has_wild = ('*' in sval) or ('%' in sval)
            pattern = self._quote_sql_string(sval.replace('*', '%'))
            expr = f"{col} LIKE '{pattern}'" if has_wild else f"{col} = '{pattern}'"
            return f"NOT ({expr})" if is_neg else expr
        # Numerics or others
        return f"{col} = {val}"

    def _build_numeric_list_predicate(self, col, pos_items, neg_items):
        """Build predicate for a numeric column list with separated positives and negatives."""
        # Positives: split into simple numbers and complex expressions
        simple_numbers = []
        complex_parts = []
        for v in pos_items:
            if isinstance(v, (int, float)):
                simple_numbers.append(v)
            elif isinstance(v, bool):
                simple_numbers.append(int(v))
            elif isinstance(v, str):
                parsed = self._parse_numeric_expr(v)
                if parsed is not None:
                    if parsed[0] == 'between':
                        complex_parts.append(f"{col} BETWEEN {parsed[1]} AND {parsed[2]}")
                    else:
                        complex_parts.append(f"{col} {parsed[1]} {parsed[2]}")
                else:
                    try:
                        num = float(v)
                        if v.strip().isdigit():
                            simple_numbers.append(int(num))
                        else:
                            simple_numbers.append(num)
                    except Exception:
                        raise InvalidQueryError(
                            f"Column '{col}' is numeric; unsupported value '{v}'. Use numbers, comparisons like '<10', or ranges like '5..10'."
                        )
            else:
                simple_numbers.append(v)

        parts = []
        if simple_numbers:
            vals = [str(sn) for sn in simple_numbers]
            parts.append(f"{col} IN (" + ", ".join(vals) + ")")
        if complex_parts:
            parts.extend(complex_parts)
        if len(parts) == 1:
            pos_expr = parts[0]
        elif len(parts) > 1:
            pos_expr = '(' + ' OR '.join(parts) + ')'
        else:
            pos_expr = ''

        # Negatives: NOT(complex) or != numeric
        neg_parts = []
        for nv in neg_items:
            parsed = self._parse_numeric_expr(nv)
            if parsed is not None:
                if parsed[0] == 'between':
                    neg_parts.append(f"NOT ({col} BETWEEN {parsed[1]} AND {parsed[2]})")
                else:
                    neg_parts.append(f"NOT ({col} {parsed[1]} {parsed[2]})")
            else:
                try:
                    num = float(nv)
                    if nv.strip().isdigit():
                        neg_parts.append(f"{col} != {int(num)}")
                    else:
                        neg_parts.append(f"{col} != {num}")
                except Exception:
                    raise InvalidQueryError(f"Column '{col}' is numeric; unsupported negated value '!{nv}'.")

        if neg_parts and pos_expr:
            return '(' + ' AND '.join(neg_parts) + ') AND ' + pos_expr
        if neg_parts:
            return ' AND '.join(neg_parts)
        return pos_expr

    def _build_string_list_predicate(self, col, pos_items, neg_items):
        """Build predicate for a non-numeric column list with separated positives and negatives."""
        pos_like_parts = []
        pos_eq_vals = []
        for v in pos_items:
            if isinstance(v, bool):
                pos_eq_vals.append(str(int(v)))
            elif isinstance(v, str):
                if ('*' in v) or ('%' in v):
                    patt = self._quote_sql_string(v.replace('*', '%'))
                    pos_like_parts.append(f"{col} LIKE '{patt}'")
                else:
                    pos_eq_vals.append("'" + self._quote_sql_string(v) + "'")
            else:
                pos_eq_vals.append(str(v))

        pos_parts = []
        if pos_eq_vals:
            pos_parts.append(f"{col} IN (" + ", ".join(pos_eq_vals) + ")")
        if pos_like_parts:
            pos_parts.extend(pos_like_parts)
        if len(pos_parts) == 1:
            pos_expr = pos_parts[0]
        elif len(pos_parts) > 1:
            pos_expr = '(' + ' OR '.join(pos_parts) + ')'
        else:
            pos_expr = ''

        neg_parts = []
        for nv in neg_items:
            if ('*' in nv) or ('%' in nv):
                patt = self._quote_sql_string(nv.replace('*', '%'))
                neg_parts.append(f"NOT ({col} LIKE '{patt}')")
            else:
                neg_parts.append(f"{col} != '" + self._quote_sql_string(nv) + "'")

        if neg_parts and pos_expr:
            return '(' + ' AND '.join(neg_parts) + ') AND ' + pos_expr
        if neg_parts:
            return ' AND '.join(neg_parts)
        return pos_expr

    def _format_criteria_conditions(self, catalog, table, criteria):
        """
        Turn a criteria dict into ADQL WHERE clause expressions, aware of column types.

        - Scalars: equality (strings quoted; booleans -> 0/1; numerics raw).
        - Strings with wildcards '*' or '%': uses LIKE (converting '*' to '%').
        - Lists/Tuples: if any string contains wildcard, build OR of LIKEs; otherwise use IN (...).
        - Numeric columns: support comparison strings ('<10', '>= 5') and ranges ('5..10', inclusive).
            Empty lists yield a false predicate (1=0).
        - Negation: a value prefixed with '!' is treated as a negated predicate. For list values, all negations are
            AND'ed together and combined with the OR of positives: (neg1 AND neg2) AND (pos1 OR pos2 ...).

        Parameters
        ----------
        criteria : dict
            Mapping of column name -> scalar or list of scalars.

        Returns
        -------
        list of str
            ADQL predicate strings (without leading WHERE/AND), suitable for joining with ' AND '.
        """
        numeric_cols = self._get_numeric_columns(catalog, table)
        conditions = []
        for key, value in criteria.items():
            # Handle list-like values => IN or OR(LIKE ...)
            if isinstance(value, (list, tuple)):
                values = list(value)
                if len(values) == 0:
                    conditions.append("1=0")
                    continue
                # Separate negatives (prefixed with '!') and positives
                neg_items = []
                pos_items = []
                for v in values:
                    if isinstance(v, str) and v.startswith('!'):
                        neg_items.append(v[1:].strip())
                    else:
                        pos_items.append(v)

                if key in numeric_cols:
                    expr = self._build_numeric_list_predicate(key, pos_items, neg_items)
                    if expr:
                        conditions.append(expr)
                else:
                    expr = self._build_string_list_predicate(key, pos_items, neg_items)
                    if expr:
                        conditions.append(expr)
            else:
                conditions.append(self._format_scalar_predicate(key, value, numeric_cols))
        return conditions

    # def _get_service_col_config(self, catalog, release='dr2', table='mean'):
    #     """
    #     For a given Catalogs.MAST catalog, return a list of all searchable columns and their descriptions.
    #     As of now, this function is exclusive to the Pan-STARRS catalog.

    #     Parameters
    #     ----------
    #     catalog : str
    #         The catalog to be queried.
    #     release : str, optional
    #         Catalog data release to query from.
    #     table : str, optional
    #         Catalog table to query from.

    #     Returns
    #     -------
    #     response : `~astropy.table.Table` that contains columns names, types, and descriptions
    #     """
    #     # Only supported for PanSTARRS currently
    #     if catalog != 'panstarrs':
    #         return

    #     service_key = (catalog, release, table)
    #     if service_key not in self._service_columns:
    #         try:
    #             # Send server request to get column list for given parameters
    #             request_url = f'{conf.catalogs_server}/api/v0.1/{catalog}/{release}/{table}/metadata.json'
    #             resp = utils._simple_request(request_url)

    #             # Parse JSON and extract necessary info
    #             results = resp.json()
    #             rows = [
    #                 (result['column_name'], result['db_type'], result['description'])
    #                 for result in results
    #             ]

    #             # Create Table with parsed data
    #             col_table = Table(rows=rows, names=('name', 'data_type', 'description'))
    #             self._service_columns[service_key] = col_table

    #         except JSONDecodeError as ex:
    #             raise JSONDecodeError(f'Failed to decode JSON response while attempting to get column list'
    #                                   f' for {catalog} catalog {table}, {release}: {ex}')
    #         except RequestException as ex:
    #             raise ConnectionError(f'Failed to connect to the server while attempting to get column list'
    #                                   f' for {catalog} catalog {table}, {release}: {ex}')
    #         except KeyError as ex:
    #             raise KeyError(f'Expected key not found in response data while attempting to get column list'
    #                            f' for {catalog} catalog {table}, {release}: {ex}')
    #         except Exception as ex:
    #             raise RuntimeError(f'An unexpected error occurred while attempting to get column list'
    #                                f' for {catalog} catalog {table}, {release}: {ex}')

    #     return self._service_columns[service_key]

    # def _validate_service_criteria(self, catalog, **criteria):
    #     """
    #     Check that criteria keyword arguments are valid column names for the service.
    #     Raises InvalidQueryError if a criteria argument is invalid.

    #     Parameters
    #     ----------
    #     catalog : str
    #         The catalog to be queried.
    #     **criteria
    #         Keyword arguments representing criteria filters to apply.

    #     Raises
    #     -------
    #     InvalidQueryError
    #         If a keyword does not match any valid column names, an error is raised that suggests the closest
    #         matching column name, if available.
    #     """
    #     # Ensure that self._service_columns is populated
    #     release = criteria.get('data_release', 'dr2')
    #     table = criteria.get('table', 'mean')
    #     col_config = self._get_service_col_config(catalog, release, table)

    #     if col_config:
    #         # Check each criteria argument for validity
    #         valid_cols = list(col_config['name']) + self._catalogs_mast_search_options
    #         for kwd in criteria.keys():
    #             col = next((name for name in valid_cols if name.lower() == kwd.lower()), None)
    #             if not col:
    #                 closest_match = difflib.get_close_matches(kwd, valid_cols, n=1)
    #                 error_msg = (
    #                     f"Filter '{kwd}' does not exist for {catalog} catalog {table}, {release}. "
    #                     f"Did you mean '{closest_match[0]}'?"
    #                     if closest_match
    #                     else f"Filter '{kwd}' does not exist for {catalog} catalog {table}, {release}."
    #                 )
    #                 raise InvalidQueryError(error_msg)

    # @class_or_instance
    # def query_region_async(self, coordinates, *, radius=0.2*u.deg, catalog="Hsc",
    #                        version=None, pagesize=None, page=None, **criteria):
    #     """
    #     Given a sky position and radius, returns a list of catalog entries.
    #     See column documentation for specific catalogs `here <https://mast.stsci.edu/api/v0/pages.html>`__.

    #     Parameters
    #     ----------
    #     coordinates : str or `~astropy.coordinates` object
    #         The target around which to search. It may be specified as a
    #         string or as the appropriate `~astropy.coordinates` object.
    #     radius : str or `~astropy.units.Quantity` object, optional
    #         Default 0.2 degrees.
    #         The string must be parsable by `~astropy.coordinates.Angle`. The
    #         appropriate `~astropy.units.Quantity` object from
    #         `~astropy.units` may also be used. Defaults to 0.2 deg.
    #     catalog : str, optional
    #         Default HSC.
    #         The catalog to be queried.
    #     version : int, optional
    #         Version number for catalogs that have versions. Default is highest version.
    #     pagesize : int, optional
    #         Default None.
    #         Can be used to override the default pagesize for (set in configs) this query only.
    #         E.g. when using a slow internet connection.
    #     page : int, optional
    #         Default None.
    #         Can be used to override the default behavior of all results being returned to obtain a
    #         specific page of results.
    #     **criteria
    #         Other catalog-specific keyword args.
    #         These can be found in the (service documentation)[https://mast.stsci.edu/api/v0/_services.html]
    #         for specific catalogs. For example, one can specify the magtype for an HSC search.
    #         For catalogs available through Catalogs.MAST (PanSTARRS), the Column Name is the keyword, and the argument
    #         should be either an acceptable value for that parameter, or a list consisting values, or  tuples of
    #         decorator, value pairs (decorator, value). In addition, columns may be used to select the return columns,
    #         consisting of a list of column names. Results may also be sorted through the query with the parameter
    #         sort_by composed of either a single Column Name to sort ASC, or a list of Column Nmaes to sort ASC or
    #         tuples of Column Name and Direction (ASC, DESC) to indicate sort order (Column Name, DESC).
    #         Detailed information of Catalogs.MAST criteria usage can
    #         be found `here <https://catalogs.mast.stsci.edu/docs/index.html>`__.

    #     Returns
    #     -------
    #     response : list of `~requests.Response`
    #     """

    #     # Put coordinates and radius into consistent format
    #     coordinates = commons.parse_coordinates(coordinates, return_frame='icrs')

    #     # if radius is just a number we assume degrees
    #     radius = coord.Angle(radius, u.deg)

    #     # basic params
    #     params = {'ra': coordinates.ra.deg,
    #               'dec': coordinates.dec.deg,
    #               'radius': radius.deg}

    #     # Determine API connection and service name
    #     if catalog.lower() in self._service_api_connection.SERVICES:
    #         self._current_connection = self._service_api_connection
    #         service = catalog

    #         # validate user criteria
    #         self._validate_service_criteria(catalog.lower(), **criteria)

    #         # adding additional user specified parameters
    #         for prop, value in criteria.items():
    #             params[prop] = value

    #     else:
    #         self._current_connection = self._portal_api_connection

    #         # valid criteria keywords
    #         valid_criteria = []

    #         # Sorting out the non-standard portal service names
    #         if catalog.lower() == "hsc":
    #             if version == 2:
    #                 service = "Mast.Hsc.Db.v2"
    #             else:
    #                 if version not in (3, None):
    #                     warnings.warn("Invalid HSC version number, defaulting to v3.", InputWarning)
    #                 service = "Mast.Hsc.Db.v3"

    #             # Hsc specific parameters (can be overridden by user)
    #             self.catalog_limit = criteria.pop('nr', 50000)
    #             valid_criteria = ['nr', 'ni', 'magtype']
    #             params['nr'] = self.catalog_limit
    #             params['ni'] = criteria.pop('ni', 1)
    #             params['magtype'] = criteria.pop('magtype', 1)

    #         elif catalog.lower() == "galex":
    #             service = "Mast.Galex.Catalog"
    #             self.catalog_limit = criteria.get('maxrecords', 50000)

    #             # galex specific parameters (can be overridden by user)
    #             valid_criteria = ['maxrecords']
    #             params['maxrecords'] = criteria.pop('maxrecords', 50000)

    #         elif catalog.lower() == "gaia":
    #             if version == 1:
    #                 service = "Mast.Catalogs.GaiaDR1.Cone"
    #             else:
    #                 if version not in (None, 2):
    #                     warnings.warn("Invalid Gaia version number, defaulting to DR2.", InputWarning)
    #                 service = "Mast.Catalogs.GaiaDR2.Cone"

    #         elif catalog.lower() == 'plato':
    #             if version in (None, 1):
    #                 service = "Mast.Catalogs.Plato.Cone"
    #             else:
    #                 warnings.warn("Invalid PLATO catalog version number, defaulting to DR1.", InputWarning)
    #                 service = "Mast.Catalogs.Plato.Cone"

    #         else:
    #             service = "Mast.Catalogs." + catalog + ".Cone"
    #             self.catalog_limit = None

    #         # additional user-specified parameters are not valid
    #         if criteria:
    #             key = next(iter(criteria))
    #             closest_match = difflib.get_close_matches(key, valid_criteria, n=1)
    #             error_msg = (
    #                 f"Filter '{key}' does not exist for catalog {catalog}. Did you mean '{closest_match[0]}'?"
    #                 if closest_match
    #                 else f"Filter '{key}' does not exist for catalog {catalog}."
    #             )
    #             raise InvalidQueryError(error_msg)

    #     # Parameters will be passed as JSON objects only when accessing the PANSTARRS API
    #     use_json = catalog.lower() == 'panstarrs'

    #     return self._current_connection.service_request_async(service, params, pagesize=pagesize, page=page,
    #                                                           use_json=use_json)

    # @class_or_instance
    # def query_object_async(self, objectname, *, radius=0.2*u.deg, catalog="Hsc",
    #                        pagesize=None, page=None, version=None, resolver=None, **criteria):
    #     """
    #     Given an object name, returns a list of catalog entries.
    #     See column documentation for specific catalogs `here <https://mast.stsci.edu/api/v0/pages.html>`__.

    #     Parameters
    #     ----------
    #     objectname : str
    #         The name of the target around which to search.
    #     radius : str or `~astropy.units.Quantity` object, optional
    #         Default 0.2 degrees.
    #         The string must be parsable by `~astropy.coordinates.Angle`.
    #         The appropriate `~astropy.units.Quantity` object from
    #         `~astropy.units` may also be used. Defaults to 0.2 deg.
    #     catalog : str, optional
    #         Default HSC.
    #         The catalog to be queried.
    #     pagesize : int, optional
    #         Default None.
    #         Can be used to override the default pagesize for (set in configs) this query only.
    #         E.g. when using a slow internet connection.
    #     page : int, optional
    #         Default None.
    #         Can be used to override the default behavior of all results being returned
    #         to obtain a specific page of results.
    #     version : int, optional
    #         Version number for catalogs that have versions. Default is highest version.
    #     resolver : str, optional
    #         The resolver to use when resolving a named target into coordinates. Valid options are "SIMBAD" and "NED".
    #         If not specified, the default resolver order will be used. Please see the
    #         `STScI Archive Name Translation Application (SANTA) <https://mastresolver.stsci.edu/Santa-war/>`__
    #         for more information. Default is None.
    #     **criteria
    #         Catalog-specific keyword args.
    #         These can be found in the `service documentation <https://mast.stsci.edu/api/v0/_services.html>`__.
    #         for specific catalogs. For example, one can specify the magtype for an HSC search.
    #         For catalogs available through Catalogs.MAST (PanSTARRS), the Column Name is the keyword, and the argument
    #         should be either an acceptable value for that parameter, or a list consisting values, or  tuples of
    #         decorator, value pairs (decorator, value). In addition, columns may be used to select the return columns,
    #         consisting of a list of column names. Results may also be sorted through the query with the parameter
    #         sort_by composed of either a single Column Name to sort ASC, or a list of Column Nmaes to sort ASC or
    #         tuples of Column Name and Direction (ASC, DESC) to indicate sort order (Column Name, DESC).
    #         Detailed information of Catalogs.MAST criteria usage can
    #         be found `here <https://catalogs.mast.stsci.edu/docs/index.html>`__.

    #     Returns
    #     -------
    #     response : list of `~requests.Response`
    #     """

    #     coordinates = utils.resolve_object(objectname, resolver=resolver)

    #     return self.query_region_async(coordinates,
    #                                    radius=radius,
    #                                    catalog=catalog,
    #                                    version=version,
    #                                    pagesize=pagesize,
    #                                    page=page,
    #                                    **criteria)

    # @class_or_instance
    # def query_criteria_async(self, catalog, *, pagesize=None, page=None, resolver=None, **criteria):
    #     """
    #     Given an set of filters, returns a list of catalog entries.
    #     See column documentation for specific catalogs `here <https://mast.stsci.edu/api/v0/pages.html>`__.

    #     Parameters
    #     ----------
    #     catalog : str
    #         The catalog to be queried.
    #     pagesize : int, optional
    #         Can be used to override the default pagesize.
    #         E.g. when using a slow internet connection.
    #     page : int, optional
    #         Can be used to override the default behavior of all results being returned to obtain
    #         one specific page of results.
    #     resolver : str, optional
    #         The resolver to use when resolving a named target into coordinates. Valid options are "SIMBAD" and "NED".
    #         If not specified, the default resolver order will be used. Please see the
    #         `STScI Archive Name Translation Application (SANTA) <https://mastresolver.stsci.edu/Santa-war/>`__
    #         for more information. Default is None.
    #     **criteria
    #         Criteria to apply. At least one non-positional criteria must be supplied.
    #         Valid criteria are coordinates, objectname, radius (as in `query_region` and `query_object`),
    #         and all fields listed in the column documentation for the catalog being queried.
    #         The Column Name is the keyword, with the argument being one or more acceptable values for that parameter,
    #         except for fields with a float datatype where the argument should be in the form [minVal, maxVal].
    #         For non-float type criteria wildcards maybe used (both * and % are considered wildcards), however
    #         only one wildcarded value can be processed per criterion.
    #         RA and Dec must be given in decimal degrees, and datetimes in MJD.
    #         For example: filters=["FUV","NUV"],proposal_pi="Ost*",t_max=[52264.4586,54452.8914]
    #         For catalogs available through Catalogs.MAST (PanSTARRS), the Column Name is the keyword, and the argument
    #         should be either an acceptable value for that parameter, or a list consisting values, or  tuples of
    #         decorator, value pairs (decorator, value). In addition, columns may be used to select the return columns,
    #         consisting of a list of column names. Results may also be sorted through the query with the parameter
    #         sort_by composed of either a single Column Name to sort ASC, or a list of Column Nmaes to sort ASC or
    #         tuples of Column Name and Direction (ASC, DESC) to indicate sort order (Column Name, DESC).
    #         Detailed information of Catalogs.MAST criteria usage can
    #         be found `here <https://catalogs.mast.stsci.edu/docs/index.html>`__.

    #     Returns
    #     -------
    #     response : list of `~requests.Response`
    #     """

    #     # Separating any position info from the rest of the filters
    #     coordinates = criteria.pop('coordinates', None)
    #     objectname = criteria.pop('objectname', None)
    #     radius = criteria.pop('radius', 0.2*u.deg)

    #     if objectname or coordinates:
    #         coordinates = utils.parse_input_location(coordinates=coordinates,
    #                                                  objectname=objectname,
    #                                                  resolver=resolver)

    #     # if radius is just a number we assume degrees
    #     radius = coord.Angle(radius, u.deg)

    #     # build query
    #     params = {}
    #     if coordinates:
    #         params["ra"] = coordinates.ra.deg
    #         params["dec"] = coordinates.dec.deg
    #         params["radius"] = radius.deg

    #     # Determine API connection, service name, and build filter set
    #     filters = None
    #     if catalog.lower() in self._service_api_connection.SERVICES:
    #         self._current_connection = self._service_api_connection
    #         service = catalog

    #         # validate user criteria
    #         self._validate_service_criteria(catalog.lower(), **criteria)

    #         if not self._current_connection.check_catalogs_criteria_params(criteria):
    #             raise InvalidQueryError("At least one non-positional criterion must be supplied.")

    #         for prop, value in criteria.items():
    #             params[prop] = value

    #     else:
    #         self._current_connection = self._portal_api_connection

    #         if catalog.lower() == "tic":
    #             service = "Mast.Catalogs.Filtered.Tic"
    #             if coordinates or objectname:
    #                 service += ".Position"
    #             service += ".Rows"  # Using the rowstore version of the query for speed
    #             column_config_name = "Mast.Catalogs.Tess.Cone"
    #             params["columns"] = "*"
    #         elif catalog.lower() == "ctl":
    #             service = "Mast.Catalogs.Filtered.Ctl"
    #             if coordinates or objectname:
    #                 service += ".Position"
    #             service += ".Rows"  # Using the rowstore version of the query for speed
    #             column_config_name = "Mast.Catalogs.Tess.Cone"
    #             params["columns"] = "*"
    #         elif catalog.lower() == "diskdetective":
    #             service = "Mast.Catalogs.Filtered.DiskDetective"
    #             if coordinates or objectname:
    #                 service += ".Position"
    #             column_config_name = "Mast.Catalogs.Dd.Cone"
    #         else:
    #             raise InvalidQueryError("Criteria query not available for {}".format(catalog))

    #         filters = self._current_connection.build_filter_set(column_config_name, service, **criteria)

    #         if not filters:
    #             raise InvalidQueryError("At least one non-positional criterion must be supplied.")
    #         params["filters"] = filters

    #     # Parameters will be passed as JSON objects only when accessing the PANSTARRS API
    #     use_json = catalog.lower() == 'panstarrs'

    #     return self._current_connection.service_request_async(service, params, pagesize=pagesize, page=page,
    #                                                           use_json=use_json)

    # @class_or_instance
    # def query_hsc_matchid_async(self, match, *, version=3, pagesize=None, page=None):
    #     """
    #     Returns all the matches for a given Hubble Source Catalog MatchID.

    #     Parameters
    #     ----------
    #     match : int or `~astropy.table.Row`
    #         The matchID or HSC entry to return matches for.
    #     version : int, optional
    #         The HSC version to match against. Default is v3.
    #     pagesize : int, optional
    #         Can be used to override the default pagesize.
    #         E.g. when using a slow internet connection.
    #     page : int, optional
    #         Can be used to override the default behavior of all results being returned to obtain
    #         one specific page of results.

    #     Returns
    #     -------
    #     response : list of `~requests.Response`
    #     """

    #     self._current_connection = self._portal_api_connection

    #     if isinstance(match, Row):
    #         match = match["MatchID"]
    #     match = str(match)  # np.int64 gives json serializer problems, so stringify right here

    #     if version == 2:
    #         service = "Mast.HscMatches.Db.v2"
    #     else:
    #         if version not in (3, None):
    #             warnings.warn("Invalid HSC version number, defaulting to v3.", InputWarning)
    #         service = "Mast.HscMatches.Db.v3"

    #     params = {"input": match}

    #     return self._current_connection.service_request_async(service, params, pagesize=pagesize, page=page)

    # @class_or_instance
    # def get_hsc_spectra_async(self, *, pagesize=None, page=None):
    #     """
    #     Returns all Hubble Source Catalog spectra.

    #     Parameters
    #     ----------
    #     pagesize : int, optional
    #         Can be used to override the default pagesize.
    #         E.g. when using a slow internet connection.
    #     page : int, optional
    #         Can be used to override the default behavior of all results being returned to obtain
    #         one specific page of results.

    #     Returns
    #     -------
    #     response : list of `~requests.Response`
    #     """

    #     self._current_connection = self._portal_api_connection

    #     service = "Mast.HscSpectra.Db.All"
    #     params = {}

    #     return self._current_connection.service_request_async(service, params, pagesize, page)

    # def download_hsc_spectra(self, spectra, *, download_dir=None, cache=True, curl_flag=False):
    #     """
    #     Download one or more Hubble Source Catalog spectra.

    #     Parameters
    #     ----------
    #     spectra : `~astropy.table.Table` or `~astropy.table.Row`
    #         One or more HSC spectra to be downloaded.
    #     download_dir : str, optional
    #        Specify the base directory to download spectra into.
    #        Spectra will be saved in the subdirectory download_dir/mastDownload/HSC.
    #        If download_dir is not specified the base directory will be '.'.
    #     cache : bool, optional
    #         Default is True. If file is found on disc it will not be downloaded again.
    #         Note: has no affect when downloading curl script.
    #     curl_flag : bool, optional
    #         Default is False.  If true instead of downloading files directly, a curl script
    #         will be downloaded that can be used to download the data files at a later time.

    #     Returns
    #     -------
    #     response : list of `~requests.Response`
    #     """

    #     # if spectra is not a Table, put it in a list
    #     if isinstance(spectra, Row):
    #         spectra = [spectra]

    #     # set up the download directory and paths
    #     if not download_dir:
    #         download_dir = '.'

    #     if curl_flag:  # don't want to download the files now, just the curl script

    #         download_file = "mastDownload_" + time.strftime("%Y%m%d%H%M%S")

    #         url_list = []
    #         path_list = []
    #         for spec in spectra:
    #             if spec['SpectrumType'] < 2:
    #                 url_list.append('https://hla.stsci.edu/cgi-bin/getdata.cgi?config=ops&dataset={0}'
    #                                 .format(spec['DatasetName']))

    #             else:
    #                 url_list.append('https://hla.stsci.edu/cgi-bin/ecfproxy?file_id={0}'
    #                                 .format(spec['DatasetName']) + '.fits')

    #             path_list.append(download_file + "/HSC/" + spec['DatasetName'] + '.fits')

    #         description_list = [""]*len(spectra)
    #         producttype_list = ['spectrum']*len(spectra)

    #         service = "Mast.Bundle.Request"
    #         params = {"urlList": ",".join(url_list),
    #                   "filename": download_file,
    #                   "pathList": ",".join(path_list),
    #                   "descriptionList": list(description_list),
    #                   "productTypeList": list(producttype_list),
    #                   "extension": 'curl'}

    #         response = self._portal_api_connection.service_request_async(service, params)
    #         bundler_response = response[0].json()

    #         local_path = os.path.join(download_dir, "{}.sh".format(download_file))
    #         self._download_file(bundler_response['url'], local_path, head_safe=True, continuation=False)

    #         status = "COMPLETE"
    #         msg = None
    #         url = None

    #         if not os.path.isfile(local_path):
    #             status = "ERROR"
    #             msg = "Curl could not be downloaded"
    #             url = bundler_response['url']
    #         else:
    #             missing_files = [x for x in bundler_response['statusList'].keys()
    #                              if bundler_response['statusList'][x] != 'COMPLETE']
    #             if len(missing_files):
    #                 msg = "{} files could not be added to the curl script".format(len(missing_files))
    #                 url = ",".join(missing_files)

    #         manifest = Table({'Local Path': [local_path],
    #                           'Status': [status],
    #                           'Message': [msg],
    #                           "URL": [url]})

    #     else:
    #         base_dir = download_dir.rstrip('/') + "/mastDownload/HSC"

    #         if not os.path.exists(base_dir):
    #             os.makedirs(base_dir)

    #         manifest_array = []
    #         for spec in spectra:

    #             if spec['SpectrumType'] < 2:
    #                 data_url = f'https://hla.stsci.edu/cgi-bin/getdata.cgi?config=ops&dataset={spec["DatasetName"]}'
    #             else:
    #                 data_url = f'https://hla.stsci.edu/cgi-bin/ecfproxy?file_id={spec["DatasetName"]}.fits'

    #             local_path = os.path.join(base_dir, f'{spec["DatasetName"]}.fits')

    #             status = "COMPLETE"
    #             msg = None
    #             url = None

    #             try:
    #                 self._download_file(data_url, local_path, cache=cache, head_safe=True)

    #                 # check file size also this is where would perform md5
    #                 if not os.path.isfile(local_path):
    #                     status = "ERROR"
    #                     msg = "File was not downloaded"
    #                     url = data_url

    #             except HTTPError as err:
    #                 status = "ERROR"
    #                 msg = "HTTPError: {0}".format(err)
    #                 url = data_url

    #             manifest_array.append([local_path, status, msg, url])

    #             manifest = Table(rows=manifest_array, names=('Local Path', 'Status', 'Message', "URL"))

    #     return manifest


Catalogs = CatalogsClass()
