# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
MAST Collections
================

This module contains methods for discovering and querying MAST catalog collections.
"""

import difflib
import math
import os
import re
import time
import warnings
from collections.abc import Iterable, Mapping
from numbers import Integral

import astropy.coordinates as coord
import astropy.units as u
import requests
from astropy.table import Row, Table
from astropy.time import Time
from astropy.utils.decorators import deprecated, deprecated_renamed_argument
from astropy.utils.exceptions import AstropyWarning

from ..exceptions import InputWarning, InvalidQueryError, NoResultsWarning
from ..utils import async_to_sync
from ..utils.class_or_instance import class_or_instance
from . import conf, utils
from .catalog_collection import CatalogCollection
from .core import MastQueryWithLogin

try:
    from regions import CircleSkyRegion, PolygonSkyRegion
    HAS_REGIONS = True
except ImportError:
    HAS_REGIONS = False

__all__ = ["Catalogs", "CatalogsClass"]


@async_to_sync
class CatalogsClass(MastQueryWithLogin):
    """
    Class for discovering and querying MAST catalog collections.
    """

    TAP_BASE_URL = conf.server + "/vo-tap/api/v0.1/"
    _collections_cache = dict()

    def __init__(self, collection=None, catalog=None):

        super().__init__()

        self._available_collections = None  # Lazy load on first request
        self._no_longer_supported_collections = ["ctl", "diskdetective", "galex", "plato"]
        self._renamed_collections = {"panstarrs": "ps1_dr2", "gaia": "gaiadr3"}

        # Default initialization of this class should not trigger network requests
        # Only set collection and catalog if explicitly provided, otherwise defer to property setters
        # which will handle defaults without network calls
        if not collection:
            self._collection = CatalogCollection("hsc")  # default collection
        else:
            self.collection = collection  # Use the setter for validation if collection is provided

        if not catalog:
            self._catalog = self._collection.default_catalog
        else:
            self.catalog = catalog  # Use the setter for validation if catalog is provided

    @property
    def collection(self):
        """
        The current MAST collection to be queried.
        """
        # Return the collection name instead of the object for easier user interaction,
        # but keep the object internally for API calls
        return self._collection.name

    @collection.setter
    def collection(self, collection):
        """
        Setter that creates a CatalogCollection object when the collection is changed and updates
        the catalog accordingly.
        """
        collection_obj = self._get_collection_obj(collection)
        self._collection = collection_obj

        # Only change catalog if not set yet or invalid for this collection
        if not hasattr(self, "_catalog") or self._catalog not in collection_obj.catalog_names:
            self._catalog = collection_obj.default_catalog

    @property
    def catalog(self):
        """
        The current catalog within the MAST collection.
        """
        return self._catalog

    @catalog.setter
    def catalog(self, catalog):
        """
        Setter that verifies that the catalog is valid for the current collection.
        """
        catalog = self._collection._verify_catalog(catalog)
        self._catalog = catalog

    @property
    def available_collections(self):
        """
        The list of available MAST catalog collections.
        """
        if self._available_collections is None:
            table = self.get_collections()
            self._available_collections = table["collection_name"].tolist()
        return self._available_collections

    @class_or_instance
    def get_collections(self):
        """
        Return a list of available collections from MAST.

        Returns
        -------
        response : `~astropy.table.Table`
            A table containing the available MAST collections.
        """
        # If already cached, use it directly
        if getattr(self, "_available_collections", None):
            return Table([self._available_collections], names=("collection_name",))

        # Otherwise, fetch from TAP service discovery, including grouped collections.
        collection_table = CatalogCollection.discover_collections()[["collection_name"]]
        return collection_table

    @class_or_instance
    def get_catalogs(self, collection=None):
        """
        For a given collection, return a list of available catalogs.

        Parameters
        ----------
        collection : str, optional
            The collection to be queried.

        Returns
        -------
        response : `~astropy.table.Table`
            A table containing the available catalogs within the specified collection.
        """
        # If no collection specified, use the class attribute
        collection_obj = self._get_collection_obj(collection) if collection else self._collection
        catalogs = collection_obj.catalogs

        # Do not expose tap_schema catalogs to users
        mask = [not str(name).startswith('tap_schema') for name in catalogs['catalog_name']]
        catalogs = catalogs[mask]

        return catalogs

    @class_or_instance
    def get_column_metadata(self, collection=None, catalog=None):
        """
        For a given collection and catalog, return metadata about the catalog's columns.

        Parameters
        ----------
        collection : str, optional
            The collection to be queried.
        catalog : str, optional
            The catalog within the collection to get metadata for.

        Returns
        -------
        response : `~astropy.table.Table`
            A table containing metadata about the specified catalog, including column names, data types,
            and descriptions.
        """
        collection_obj, catalog = self._parse_inputs(collection, catalog)
        return collection_obj.get_catalog_metadata(catalog).column_metadata

    @class_or_instance
    def get_crossmatch_catalogs(self, collection=None, catalog=None):
        """
        List other catalogs that can be crossmatched with the specified catalog.

        Parameters
        ----------
        collection : str, optional
            Collection of the reference catalog. Defaults to the instance's collection.
        catalog : str, optional
            Reference catalog, resolved as in `get_catalogs` and `crossmatch`.

        Returns
        -------
        `~astropy.table.Table`
            Columns ``collection_name`` and ``catalog_name`` for catalogs on the same
            TAP endpoint that have identifiable RA/Dec columns. These are candidates;
            sharing an endpoint and having coordinates does not guarantee JOIN support.
            Rows are unique and sorted by collection name, then catalog name.

        Notes
        -----
        This method fetches catalog and column metadata, which may require several
        requests. Metadata is cached for subsequent calls.
        """
        collection_obj, catalog = self._parse_inputs(collection, catalog)
        discovered = CatalogCollection.discover_collections()
        rows = set()

        def display_name(name):
            # Remove the "mast_cat_" prefix for display purposes if applicable
            name = str(name)
            prefix = "mast_cat_"
            if collection_obj.parent_collection == "mast_catalogs" and name.casefold().startswith(prefix):
                return name[len(prefix):]
            return name

        for row in discovered:
            if row["parent_collection"] != collection_obj.parent_collection:
                continue
            candidate = self._get_collection_obj(str(row["collection_name"]))
            for name in self.get_catalogs(candidate.name)["catalog_name"]:
                name = str(name)
                if display_name(name).casefold() == display_name(catalog).casefold():
                    continue
                metadata = candidate.get_catalog_metadata(name)
                if metadata.ra_column and metadata.dec_column:
                    rows.add((display_name(candidate.name), display_name(name)))
        return Table(rows=sorted(rows), names=("collection_name", "catalog_name"), dtype=(str, str))

    def supports_spatial_queries(self, collection=None, catalog=None):
        """
        Check if a given collection and catalog support spatial queries.

        Parameters
        ----------
        collection : str, optional
            The collection to be queried.
        catalog : str, optional
            The catalog within the collection to check.

        Returns
        -------
        bool
            True if the specified catalog supports spatial queries, False otherwise.
        """
        collection_obj, catalog = self._parse_inputs(collection, catalog)
        return collection_obj.get_catalog_metadata(catalog).supports_spatial_queries

    @class_or_instance
    @deprecated_renamed_argument(
        "version",
        None,
        since="0.4.13",
        message="The `version` argument is deprecated and "
        "will be removed in a future release. Please use `collection` and `catalog` instead.",
    )
    @deprecated_renamed_argument(
        "pagesize",
        None,
        since="0.4.13",
        message="The `pagesize` argument is deprecated "
        "and will be removed in a future release. Please use `limit` instead.",
    )
    @deprecated_renamed_argument(
        "page",
        None,
        since="0.4.13",
        message="The `page` argument is deprecated "
        "and will be removed in a future release. Please use `offset` instead.",
    )
    @deprecated_renamed_argument("objectname", "object_name", since="0.4.13")
    def query_criteria(
        self,
        collection=None,
        *,
        catalog=None,
        coordinates=None,
        region=None,
        object_name=None,
        radius=0.2 * u.deg,
        resolver=None,
        limit=5000,
        offset=0,
        count_only=False,
        select_cols=None,
        sort_by=None,
        sort_desc=False,
        filters=None,
        run_async=False,
        return_adql=False,
        version=None,
        pagesize=None,
        page=None,
        **criteria,
    ):
        """
        Query a MAST catalog from a given collection using criteria filters. To return columns for a given
        collection and catalog, use `~astroquery.mast.CatalogsClass.get_column_metadata`.

        Parameters
        ----------
        collection : str, optional
            The collection to be queried. If None, uses the instance's `collection` attribute.
        catalog : str, optional
            The catalog within the collection to query. If None, uses the instance's `catalog` attribute.
        coordinates : str or `~astropy.coordinates` object, optional
            The target around which to search. It may be specified as a string (e.g., '350 -80') or as an
            Astropy coordinates object.
        region : str | iterable | `~regions.CircleSkyRegion` | `~regions.PolygonSkyRegion`, optional
            The region to search within. It may be specified as a STC-S POLYGON or CIRCLE string
            (e.g., 'CIRCLE 350 -80 0.2'), an iterable of coordinate pairs, or as an
            `~regions.CircleSkyRegion` or `~regions.PolygonSkyRegion`.
        object_name : str, optional
            The name of the object to resolve and search around.
        radius : str or `~astropy.units.Quantity` object, optional
            The search radius around the target coordinates or object. Default 0.2 degrees.
        resolver : str, optional
            The name resolver service to use when resolving ``object_name``.
        limit : int, optional
            The maximum number of results to return. Default is 5000.
        offset : int, optional
            The number of rows to skip before starting to return rows. Default is 0.
        count_only : bool, optional
            If True, only return the count of matching records instead of the records themselves. Default is False.
        select_cols : list of str or `CatalogColumn`, optional
            Column names to include in the result. `CatalogColumn` references must belong to the queried catalog.
            If None or empty, all columns are returned.
        sort_by : str, `CatalogColumn`, or list of str or `CatalogColumn`, optional
            Column name(s) to sort the results by.
        sort_desc : bool or list of bool, optional
            Indicates whether to sort in descending order for each column in ``sort_by``. If a single bool,
            applies to all columns. If a list, must match length of ``sort_by``. Default is False (ascending order).
        filters : dict, optional
            Criteria as a mapping from column names or `CatalogColumn` references to values. Use this option when the
            name of a column conflicts with a named parameter of this method. References must belong to the queried
            catalog.
        run_async : bool, optional
            If True, run the query in asynchronous mode. This mode is more robust and preferable
            for long-running queries. If you encounter timeouts or connection issues with large queries,
            set this to True. Default is False (synchronous mode).
        return_adql : bool, optional
            If True, return the ADQL query string instead of executing the query. Default is False.
            When False, the ADQL query string is also returned in the metadata of the result table.
        version : str, optional
            Deprecated. The version argument is no longer used. Please use ``collection`` and ``catalog`` instead.
        pagesize : int, optional
            Deprecated. The pagesize argument is no longer used. Please use ``limit`` instead.
        page : int, optional
            Deprecated. The page argument is no longer used. Please use ``offset`` instead.
        **criteria
            Keyword arguments representing criteria filters to apply.

            Criteria syntax:

            - Strings support wildcards using '*' (converted to SQL '%') and '%'.
            - Lists are combined with OR for positive values; empty lists yield no matches.
            - Numeric columns support comparison operators ('<', '<=', '>', '>=') and inclusive ranges using
                the syntax 'low..high' (e.g., '5..10'). Mixed lists of numbers and comparisons are OR-combined.
            - Negation: Prefix any value with '!' to negate that predicate. For list inputs, all negated values
                for the same column are AND-combined, then ANDed with the OR of the positive values:
                (neg1 AND neg2 AND ...) AND (pos1 OR pos2 OR ...).
            - Temporal columns can be specified as strings in a recognized date/time format (e.g., 'YYYY', 'YYYY-MM-DD',
              'YYYY-MM-DD hh:mm:ss', etc.), `~astropy.time.Time` objects, or `datetime` objects. The same comparison
              operators and range syntax as numeric columns can be used to filter temporal columns
              based on date/time values.

            Examples:

            - file_suffix=['A', 'B', '!C'] -> (file_suffix != 'C') AND (file_suffix IN ('A', 'B'))
            - size=['!14400', '<20000'] -> (size != 14400) AND (size < 20000)

        Returns
        -------
        response : `~astropy.table.Table`
            A table containing the query results.
        """
        # Parse pagination params
        limit, offset = self._parse_legacy_pagination(limit, offset, pagesize, page)

        # Should not specify both region and coordinates
        if coordinates and region:
            raise InvalidQueryError("Specify either `region` or `coordinates`, not both.")

        # Should not specify both region and object_name
        if object_name and region:
            raise InvalidQueryError("Specify either `region` or `object_name`, not both.")

        collection_obj, catalog = self._parse_inputs(collection, catalog)
        from .catalog import CatalogColumn

        def resolve_column(column):
            """Resolve a column to its name, ensuring it belongs to the queried catalog."""
            if not isinstance(column, CatalogColumn):
                return column
            owner = column.catalog
            if (
                owner.collection_name.casefold() != collection_obj.name.casefold()
                or owner.catalog_name.casefold() != catalog.casefold()
            ):
                raise InvalidQueryError("Column references must belong to the queried Catalog.")
            return column.name

        # Normalize select_cols and sort_by to ensure consistent handling of column references
        if select_cols is not None:
            select_cols = self._normalize_columns(
                select_cols, resolve_column, CatalogColumn, scalar_string_as_list=True,
            )
        if sort_by is not None:
            sort_by = self._normalize_columns(
                sort_by, resolve_column, CatalogColumn, scalar_string_as_list=True,
            )

        # Normalize criteria keys to ensure consistent handling of column references
        normalized_filters = {}
        if filters is not None:
            if not isinstance(filters, Mapping):
                raise InvalidQueryError("`filters` must be a mapping of columns to criteria.")
            seen_filters = set()
            for column, value in filters.items():
                column = resolve_column(column)
                if not isinstance(column, str):
                    raise InvalidQueryError("Filter columns must be strings or CatalogColumn references.")
                if column.casefold() in seen_filters:
                    raise InvalidQueryError(f"Criteria column '{column}' was specified more than once.")
                seen_filters.add(column.casefold())
                normalized_filters[column] = value

        # Check for conflicts between named parameters and filters dict
        if criteria and normalized_filters:
            overlap = set(k.lower() for k in criteria) & set(k.lower() for k in normalized_filters)
            if overlap:
                raise InvalidQueryError(
                    f"Criteria specified both as keyword arguments and in 'filters' for columns: "
                    f"{', '.join(sorted(overlap))}"
                )

        # Merge criteria from named parameters and filters dict
        search_criteria = {}
        search_criteria.update(criteria)
        if normalized_filters:
            search_criteria.update(normalized_filters)

        # Validate sort_by columns together with criteria keys so all column checks go through one path
        validation_criteria = dict(search_criteria)
        if sort_by and not count_only:
            sort_by = [sort_by] if isinstance(sort_by, str) else list(sort_by)
            validation_criteria.update({col: True for col in sort_by})

        collection_obj._verify_criteria(catalog, **validation_criteria)
        catalog_metadata = collection_obj.get_catalog_metadata(catalog)
        column_metadata = catalog_metadata.column_metadata
        columns = "*" if not select_cols else self._parse_select_cols(select_cols, column_metadata)

        adql = (
            f"SELECT TOP {limit} {columns} FROM {catalog.lower()} "
            if not count_only
            else f"SELECT TOP 1 COUNT(*) AS count_all FROM {catalog.lower()} "
        )
        has_where = False
        if region or coordinates or object_name:
            # Check if the catalog supports spatial queries
            if not catalog_metadata.supports_spatial_queries:
                raise InvalidQueryError(
                    f"Catalog '{catalog}' in collection '{collection_obj.name}' does not support spatial queries."
                )

            # Positional query
            adql_region = ""
            if region:
                adql_region = self._create_adql_region(region)
            if object_name or coordinates:  # Cone search
                coordinates = utils.parse_input_location(
                    coordinates=coordinates, object_name=object_name, resolver=resolver
                )
                radius = coord.Angle(radius, u.deg)  # If radius is just a number we assume degrees
                adql_region = f"CIRCLE('ICRS', {coordinates.ra.deg}, {coordinates.dec.deg}, {radius.to(u.deg).value})"

            region_types = ["POLYGON", "CIRCLE"]
            for region_type in region_types:
                if region_type in adql_region and region_type not in collection_obj.supported_adql_functions:
                    raise InvalidQueryError(
                        f"Catalog '{catalog}' in collection '{collection_obj.name}' "
                        f"does not support ADQL region type '{region_type}'."
                    )

            # Get RA/Dec column names
            ra_col = catalog_metadata.ra_column
            dec_col = catalog_metadata.dec_column
            adql += f"WHERE CONTAINS(POINT('ICRS', {ra_col}, {dec_col}), {adql_region}) = 1 "
            has_where = True

        # Add additional constraints
        if search_criteria:
            conditions = self._format_criteria_conditions(collection_obj, catalog, search_criteria)
            if has_where:
                adql += "AND " + " AND ".join(conditions)
            else:
                adql += "WHERE " + " AND ".join(conditions)

        # Add sorting if specified
        if sort_by:
            # Add ORDER BY clause
            if isinstance(sort_desc, bool):
                sort_desc = [sort_desc]

            if len(sort_desc) not in (1, len(sort_by)):
                raise InvalidQueryError("Length of 'sort_desc' must be 1 or equal to length of 'sort_by'.")

            if len(sort_desc) == 1:
                sort_desc = sort_desc * len(sort_by)

            order_parts = [
                f"{col} {'DESC' if desc else 'ASC'}"
                for col, desc in zip(sort_by, sort_desc)
            ]

            adql += " ORDER BY " + ", ".join(order_parts)

        # Add offset
        if offset:
            adql += f" OFFSET {offset}"

        if return_adql:
            return adql

        # Execute the query
        result_table = collection_obj.run_tap_query(adql, run_async=run_async)

        # Add ADQL to table metadata for reference
        result_table.meta["adql_query"] = adql

        if len(result_table) == 0:
            warnings.warn("The query returned no results.", NoResultsWarning)

        if count_only:
            return int(result_table["count_all"][0])
        else:
            # TODO: Add schema browser URL to the result table metadata when available
            result_table.meta["collection"] = collection_obj.name
            result_table.meta["catalog"] = catalog
        return result_table

    @class_or_instance
    @deprecated_renamed_argument(
        "version",
        None,
        since="0.4.13",
        message="The `version` argument is deprecated and "
        "will be removed in a future release. Please use `collection` and `catalog` instead.",
    )
    @deprecated_renamed_argument(
        "pagesize",
        None,
        since="0.4.13",
        message="The `pagesize` argument is deprecated "
        "and will be removed in a future release. Please use `limit` instead.",
    )
    @deprecated_renamed_argument(
        "page",
        None,
        since="0.4.13",
        message="The `page` argument is deprecated "
        "and will be removed in a future release. Please use `offset` instead.",
    )
    def query_region(
        self,
        coordinates=None,
        *,
        radius=0.2 * u.deg,
        region=None,
        collection=None,
        catalog=None,
        limit=5000,
        offset=0,
        count_only=False,
        select_cols=None,
        sort_by=None,
        sort_desc=False,
        filters=None,
        run_async=False,
        return_adql=False,
        version=None,
        pagesize=None,
        page=None,
        **criteria,
    ):
        """
        Query for MAST catalog entries within a specified region using criteria filters. To return columns for a given
        collection and catalog, use `~astroquery.mast.CatalogsClass.get_column_metadata`.

        Parameters
        ----------
        coordinates : str or `~astropy.coordinates` object, optional
            The target around which to search. It may be specified as a string (e.g., '350 -80') or as an
            Astropy coordinates object.
        radius : str or `~astropy.units.Quantity` object, optional
            The search radius around the target coordinates or object. Default 0.2 degrees.
        region : str | iterable | `~regions.CircleSkyRegion` | `~regions.PolygonSkyRegion`, optional
            The region to search within. It may be specified as a STC-S POLYGON or CIRCLE string
            (e.g., 'CIRCLE 350 -80 0.2'), an iterable of coordinate pairs, or as an
            `~regions.CircleSkyRegion` or `~regions.PolygonSkyRegion`.
        collection : str, optional
            The collection to be queried. If None, uses the instance's `collection` attribute.
        catalog : str, optional
            The catalog within the collection to query. If None, uses the instance's `catalog` attribute.
        limit : int, optional
            The maximum number of results to return. Default is 5000.
        offset : int, optional
            The number of rows to skip before starting to return rows. Default is 0.
        count_only : bool, optional
            If True, only return the count of matching records instead of the records themselves. Default is False.
        select_cols : list of str, optional
            List of column names to include in the result. If None or empty, all columns are returned.
        sort_by : str or list of str, optional
            Column name(s) to sort the results by.
        sort_desc : bool or list of bool, optional
            Indicates whether to sort in descending order for each column in ``sort_by``. If a single bool,
            applies to all columns. If a list, must match length of ``sort_by``. Default is False (ascending order).
        filters : dict, optional
            Another parameter to specify criteria filters as a dictionary. Use this option when the name of a column
            conflicts with a named parameter of this method.
        run_async : bool, optional
            If True, run the query in asynchronous mode. This mode is more robust and preferable
            for long-running queries. If you encounter timeouts or connection issues with large queries,
            set this to True. Default is False (synchronous mode).
        return_adql : bool, optional
            If True, return the ADQL query string instead of executing the query. Default is False.
            When False, the ADQL query string is also returned in the metadata of the result table.
        version : str, optional
            Deprecated. The version argument is no longer used. Please use ``collection`` and ``catalog`` instead.
        pagesize : int, optional
            Deprecated. The pagesize argument is no longer used. Please use ``limit`` instead.
        page : int, optional
            Deprecated. The page argument is no longer used. Please use ``offset`` instead.
        **criteria
            Keyword arguments representing criteria filters to apply.

            Criteria syntax:

            - Strings support wildcards using '*' (converted to SQL '%') and '%'.
            - Lists are combined with OR for positive values; empty lists yield no matches.
            - Numeric columns support comparison operators ('<', '<=', '>', '>=') and inclusive ranges using
                the syntax 'low..high' (e.g., '5..10'). Mixed lists of numbers and comparisons are OR-combined.
            - Negation: Prefix any value with '!' to negate that predicate. For list inputs, all negated values
                for the same column are AND-combined, then ANDed with the OR of the positive values:
                (neg1 AND neg2 AND ...) AND (pos1 OR pos2 OR ...).
            - Temporal columns can be specified as strings in a recognized date/time format (e.g., 'YYYY', 'YYYY-MM-DD',
              'YYYY-MM-DD hh:mm:ss', etc.), `~astropy.time.Time` objects, or `datetime` objects. The same comparison
              operators and range syntax as numeric columns can be used to filter temporal columns
              based on date/time values.

            Examples:

            - file_suffix=['A', 'B', '!C'] -> (file_suffix != 'C') AND (file_suffix IN ('A', 'B'))
            - size=['!14400', '<20000'] -> (size != 14400) AND (size < 20000)

        Returns
        -------
        response : `~astropy.table.Table`
            A table containing the query results.
        """
        # Must specify one of region or coordinates
        if region is None and coordinates is None:
            raise InvalidQueryError(
                "Must specify either `region` or `coordinates`. For non-positional queries, "
                "use `Catalogs.query_criteria`."
            )

        # Parse pagination params
        limit, offset = self._parse_legacy_pagination(limit, offset, pagesize, page)

        return self.query_criteria(
            collection=collection,
            catalog=catalog,
            coordinates=coordinates,
            region=region,
            radius=radius,
            limit=limit,
            offset=offset,
            count_only=count_only,
            select_cols=select_cols,
            sort_by=sort_by,
            sort_desc=sort_desc,
            filters=filters,
            run_async=run_async,
            return_adql=return_adql,
            **criteria,
        )

    @class_or_instance
    @deprecated_renamed_argument(
        "version",
        None,
        since="0.4.13",
        message="The `version` argument is deprecated and "
        "will be removed in a future release. Please use `collection` and `catalog` instead.",
    )
    @deprecated_renamed_argument(
        "pagesize",
        None,
        since="0.4.13",
        message="The `pagesize` argument is deprecated "
        "and will be removed in a future release. Please use `limit` instead.",
    )
    @deprecated_renamed_argument(
        "page",
        None,
        since="0.4.13",
        message="The `page` argument is deprecated "
        "and will be removed in a future release. Please use `offset` instead.",
    )
    @deprecated_renamed_argument("objectname", "object_name", since="0.4.13")
    def query_object(
        self,
        object_name,
        *,
        radius=0.2 * u.deg,
        collection=None,
        catalog=None,
        resolver=None,
        limit=5000,
        offset=0,
        count_only=False,
        select_cols=None,
        sort_by=None,
        sort_desc=False,
        filters=None,
        run_async=False,
        return_adql=False,
        version=None,
        pagesize=None,
        page=None,
        **criteria,
    ):
        """
        Query for MAST catalog entries around a specified object name using criteria filters. To return columns
        for a given collection and catalog, use `~astroquery.mast.CatalogsClass.get_column_metadata`.

        Parameters
        ----------
        object_name : str, optional
            The name of the object to resolve and search around.
        radius : str or `~astropy.units.Quantity` object, optional
            The search radius around the target coordinates or object. Default 0.2 degrees.
        collection : str, optional
            The collection to be queried. If None, uses the instance's `collection` attribute.
        catalog : str, optional
            The catalog within the collection to query. If None, uses the instance's `catalog` attribute.
        resolver : str, optional
            The name resolver service to use when resolving ``object_name``.
        limit : int, optional
            The maximum number of results to return. Default is 5000.
        offset : int, optional
            The number of rows to skip before starting to return rows. Default is 0.
        count_only : bool, optional
            If True, only return the count of matching records instead of the records themselves. Default is False.
        select_cols : list of str, optional
            List of column names to include in the result. If None or empty, all columns are returned.
        sort_by : str or list of str, optional
            Column name(s) to sort the results by.
        sort_desc : bool or list of bool, optional
            Indicates whether to sort in descending order for each column in ``sort_by``. If a single bool,
            applies to all columns. If a list, must match length of ``sort_by``. Default is False (ascending order).
        filters : dict, optional
            Another parameter to specify criteria filters as a dictionary. Use this option when the name of a column
            conflicts with a named parameter of this method.
        run_async : bool, optional
            If True, run the query in asynchronous mode. This mode is more robust and preferable
            for long-running queries. If you encounter timeouts or connection issues with large queries,
            set this to True. Default is False (synchronous mode).
        return_adql : bool, optional
            If True, return the ADQL query string instead of executing the query. Default is False.
            When False, the ADQL query string is also returned in the metadata of the result table.
        version : str, optional
            Deprecated. The version argument is no longer used. Please use ``collection`` and ``catalog`` instead.
        pagesize : int, optional
            Deprecated. The pagesize argument is no longer used. Please use ``limit`` instead.
        page : int, optional
            Deprecated. The page argument is no longer used. Please use ``offset`` instead.
        **criteria
            Keyword arguments representing criteria filters to apply.

            Criteria syntax:

            - Strings support wildcards using '*' (converted to SQL '%') and '%'.
            - Lists are combined with OR for positive values; empty lists yield no matches.
            - Numeric columns support comparison operators ('<', '<=', '>', '>=') and inclusive ranges using
                the syntax 'low..high' (e.g., '5..10'). Mixed lists of numbers and comparisons are OR-combined.
            - Negation: Prefix any value with '!' to negate that predicate. For list inputs, all negated values
                for the same column are AND-combined, then ANDed with the OR of the positive values:
                (neg1 AND neg2 AND ...) AND (pos1 OR pos2 OR ...).
            - Temporal columns can be specified as strings in a recognized date/time format (e.g., 'YYYY', 'YYYY-MM-DD',
              'YYYY-MM-DD hh:mm:ss', etc.), `~astropy.time.Time` objects, or `datetime` objects. The same comparison
              operators and range syntax as numeric columns can be used to filter temporal columns
              based on date/time values.

            Examples:

            - file_suffix=['A', 'B', '!C'] -> (file_suffix != 'C') AND (file_suffix IN ('A', 'B'))
            - size=['!14400', '<20000'] -> (size != 14400) AND (size < 20000)

        Returns
        -------
        response : `~astropy.table.Table`
            A table containing the query results.
        """
        # Parse pagination params
        limit, offset = self._parse_legacy_pagination(limit, offset, pagesize, page)

        return self.query_criteria(
            collection=collection,
            catalog=catalog,
            object_name=object_name,
            radius=radius,
            resolver=resolver,
            limit=limit,
            offset=offset,
            count_only=count_only,
            select_cols=select_cols,
            sort_by=sort_by,
            sort_desc=sort_desc,
            filters=filters,
            run_async=run_async,
            return_adql=return_adql,
            **criteria,
        )

    @class_or_instance
    def crossmatch(
        self,
        collection_1=None,
        catalog_1=None,
        collection_2=None,
        catalog_2=None,
        *,
        catalogs=None,
        alias_1=None,
        alias_2=None,
        include_separation=True,
        coordinates=None,
        region=None,
        object_name=None,
        search_radius=0.2 * u.deg,
        match_radius=3 * u.arcsec,
        resolver=None,
        limit=5000,
        offset=0,
        count_only=False,
        select_cols=None,
        sort_by=None,
        sort_desc=False,
        filters=None,
        run_async=False,
        return_adql=False,
        **criteria,
    ):
        """
        Crossmatch two MAST catalogs on sky position within a region.

        All pairs with separation strictly less than ``match_radius`` are returned,
        not just the nearest match. A source may therefore appear in multiple rows.
        Spatial constraints select rows from the first catalog only; matches in the
        second catalog may lie just outside the search region.

        Parameters
        ----------
        collection_1 : str or `Catalog`, optional
            The collection of the first catalog to crossmatch, or a `Catalog` object that identifies both
            collection and catalog. If None, uses the instance's `collection` attribute.
        catalog_1 : str or `Catalog`, optional
            The first catalog to crossmatch. A `Catalog` object supplies both its collection and catalog;
            if ``collection_1`` is also supplied, it must match. If None, uses the default catalog for
            ``collection_1`` (or the instance's `catalog` attribute if ``collection_1`` is also None).
        collection_2 : str or `Catalog`, optional
            The collection of the second catalog to crossmatch, or a `Catalog` object that identifies both
            collection and catalog. If None, uses the instance's `collection` attribute.
        catalog_2 : str or `Catalog`, optional
            The second catalog to crossmatch, resolved the same way as ``catalog_1``.
        catalogs : sequence of two `Catalog` objects, collection names, or (str, str) pairs, optional
            Alternative to the four collection/catalog arguments. Each target is a
            `Catalog` object, collection name (using its default catalog), or a
            ``(collection, catalog)`` pair. For example, ``catalogs=(gaia, tic)`` or
            ``catalogs=('gaiadr3', 'tic_v82')``. Cannot be combined
            with ``collection_1``, ``catalog_1``, ``collection_2``, or ``catalog_2``.
        alias_1, alias_2 : str, optional
            Custom table aliases. Defaults to collection names, with ``_1`` and ``_2``
            appended when both collections match. The stable shorthand aliases
            ``cat1`` and ``cat2`` are also accepted in column selections, sorting,
            and filters, regardless of the chosen aliases; neither can be used as a
            custom alias, regardless of case.
        include_separation : bool, optional
            Include a ``separation`` column in arcseconds. Default True. Ignored for
            ``count_only``. When included, it can also be used in ``sort_by``.
        coordinates : str or `~astropy.coordinates` object, optional
            The target around which to search. It may be specified as a string (e.g., '350 -80') or as an
            Astropy coordinates object. None of ``coordinates``, ``region``, or ``object_name`` are required;
            if none are given, the crossmatch runs against the full extent of both catalogs (subject to
            ``filters``/``**criteria``), which may be slow.
        region : str | iterable | `~regions.CircleSkyRegion` | `~regions.PolygonSkyRegion`, optional
            The region to search within. It may be specified as a STC-S POLYGON or CIRCLE string
            (e.g., 'CIRCLE 350 -80 0.2'), an iterable of coordinate pairs, or as an
            `~regions.CircleSkyRegion` or `~regions.PolygonSkyRegion`.
        object_name : str, optional
            The name of the object to resolve and search around.
        search_radius : str or `~astropy.units.Quantity` object, optional
            The search radius around the target coordinates or object used to select rows from the first
            catalog. Only used if ``coordinates`` or ``object_name`` is given. Default 0.2 degrees.
            The maximum is 30 arcminutes. Bare numbers are interpreted as degrees.
        match_radius : str or `~astropy.units.Quantity` object, optional
            The maximum separation allowed between a row in the first catalog and a row in the second catalog
            for them to be considered a match. Default 3 arcsec; maximum 5 arcseconds.
        resolver : str, optional
            The name resolver service to use when resolving ``object_name``.
        limit : int, optional
            The maximum number of results to return. Default is 5000.
        offset : int, optional
            The number of rows to skip before starting to return rows. Default is 0.
        count_only : bool, optional
            If True, only return the count of matching records instead of the records themselves. Default is False.
        select_cols : str, list of str, or `CatalogColumn`, optional
            List of column names to include in the result. If a column name exists in both catalogs, it must
            be qualified with the owning collection name (e.g. ``'gaiadr3.ra'``). Strings may contain
            comma-separated column names. If None or empty, all columns from both catalogs are returned.
            ``cat1.ra`` and ``cat2.ra`` are always
            accepted, as are underscore-qualified forms such as ``cat1_ra`` and
            ``gaiadr3_ra``. Real catalog column names take precedence over underscore
            shorthand; use dotted qualification if they collide. `CatalogColumn` references
            are qualified according to their owning `Catalog` object.
            Result columns are named ``<alias>.<column>`` to identify their
            source and avoid duplicate names. ``separation`` is added separately.
        sort_by : str or list of str, optional
            Column name(s) to sort the results by. Ambiguous column names must be qualified as in ``select_cols``.
            Both dotted and underscore-qualified forms are accepted, and may be mixed.
        sort_desc : bool or list of bool, optional
            Indicates whether to sort in descending order for each column in ``sort_by``. If a single bool,
            applies to all columns. If a list, must match length of ``sort_by``. Default is False (ascending order).
        filters : dict, optional
            Another parameter to specify criteria filters as a dictionary. Use this option when the name of a column
            conflicts with a named parameter of this method.
        run_async : bool, optional
            If True, run the query in asynchronous mode. This mode is more robust and preferable
            for long-running queries. Default is False (synchronous mode).
        return_adql : bool, optional
            If True, return the ADQL query string instead of executing the query. Default is False.
        **criteria
            Keyword arguments representing criteria filters to apply. Ambiguous column names must be qualified
            with a table alias using ``alias.column`` or ``alias_column``, for example
            ``cat1_bmag=10`` or ``filters={'tic_v82_bmag': 10}``. Stable ``cat1``/``cat2``
            and the actual table aliases are accepted. Real catalog column names take
            precedence over underscore shorthand; use dotted qualification to resolve
            such a collision. Do not specify the same column in multiple forms.

        Returns
        -------
        response : `~astropy.table.Table`
            A table containing the crossmatched query results, an integer if
            ``count_only=True``, or an ADQL string if ``return_adql=True``.
        """
        # Validate and resolve the crossmatch targets before proceeding
        resolved_targets = self._resolve_crossmatch_targets(
            collection_1, catalog_1, collection_2, catalog_2, catalogs
        )
        (collection_1, catalog_1, catalog_object_1), (collection_2, catalog_2, catalog_object_2) = resolved_targets

        # Ensure that only one of coordinates, region, or object_name is specified
        if sum(value is not None for value in (coordinates, region, object_name)) > 1:
            raise InvalidQueryError("Specify only one of `region`, `coordinates`, or `object_name`.")

        # Validate that limit and offset are integers greater than or equal to their minimum values
        for name, value, minimum in (("limit", limit, 1), ("offset", offset, 0)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise InvalidQueryError(f"`{name}` must be an integer greater than or equal to {minimum}.")

        # Parse and validate the match radius
        match_radius_deg = self._parse_crossmatch_radius(match_radius, "match_radius")

        # If coordinates or object_name are specified, parse and validate the search radius
        if coordinates is not None or object_name is not None:
            search_radius_deg = self._parse_crossmatch_radius(search_radius, "search_radius")

        # Verify that the provided aliases are valid ADQL identifiers
        for name, alias in (("alias_1", alias_1), ("alias_2", alias_2)):
            if alias is not None and (not isinstance(alias, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", alias)):
                raise InvalidQueryError(f"`{name}` must be an ADQL identifier starting with a letter.")

        # Any catalog left unspecified falls back to the instance's current collection/catalog
        if catalog_object_1 is None:
            collection_1_obj, catalog_1 = self._parse_inputs(collection_1, catalog_1)
        else:
            collection_1_obj = catalog_object_1._query._collection
            catalog_1 = catalog_object_1.catalog_name
        if catalog_object_2 is None:
            collection_2_obj, catalog_2 = self._parse_inputs(collection_2, catalog_2)
        else:
            collection_2_obj = catalog_object_2._query._collection
            catalog_2 = catalog_object_2.catalog_name

        # Ensure that the two catalogs being crossmatched are not the same
        if (
            (collection_1_obj.name.strip().casefold(), catalog_1.strip().casefold())
            == (collection_2_obj.name.strip().casefold(), catalog_2.strip().casefold())
        ):
            raise InvalidQueryError(
                f"Cannot crossmatch catalog '{catalog_1}' with itself. Choose two different catalogs."
            )

        if (
            collection_1_obj.parent_collection == "mast_catalogs"
            and collection_2_obj.parent_collection == "roman_catalogs"
        ):
            # Swap the catalogs so that the Roman catalog is always on the left
            collection_1_obj, catalog_1, alias_1, collection_2_obj, catalog_2, alias_2 = (
                collection_2_obj,
                catalog_2,
                alias_2,
                collection_1_obj,
                catalog_1,
                alias_1,
            )
            catalog_object_1, catalog_object_2 = catalog_object_2, catalog_object_1
        elif (
            collection_1_obj.parent_collection != collection_2_obj.parent_collection
            and not (
                collection_1_obj.parent_collection == "roman_catalogs"
                and collection_2_obj.parent_collection == "mast_catalogs"
            )
        ):
            # Reject catalogs on different TAP endpoints outside the MAST/Roman case.
            raise InvalidQueryError(
                f"Cannot crossmatch '{collection_1_obj.name}' and '{collection_2_obj.name}': they are served by "
                f"different TAP endpoints ('{collection_1_obj.parent_collection}' and "
                f"'{collection_2_obj.parent_collection}'). Only catalogs served by the same TAP endpoint "
                "can be crossmatched. Use `get_crossmatch_catalogs(collection_1, catalog_1)` to find alternatives."
            )

        meta1 = collection_1_obj.get_catalog_metadata(catalog_1)
        meta2 = collection_2_obj.get_catalog_metadata(catalog_2)

        # Ensure that both catalogs have RA/Dec columns for crossmatching
        if not (meta1.ra_column and meta1.dec_column and meta2.ra_column and meta2.dec_column):
            raise InvalidQueryError(
                f"Both '{catalog_1}' and '{catalog_2}' must have identifiable RA/Dec columns to be crossmatched."
            )

        # Aliases used to qualify ambiguous column names in the JOIN and ensure they are distinct
        default_alias1, default_alias2 = collection_1_obj.name, collection_2_obj.name
        if default_alias1 == default_alias2:
            default_alias1, default_alias2 = f"{default_alias1}_1", f"{default_alias2}_2"
        alias_1 = alias_1 or default_alias1
        alias_2 = alias_2 or default_alias2
        if any(alias.casefold() in {"cat1", "cat2"} for alias in (alias_1, alias_2)):
            raise InvalidQueryError("Aliases `cat1` and `cat2` are reserved and cannot be used for `alias_1` "
                                    "or `alias_2`.")
        if alias_1.casefold() == alias_2.casefold():
            raise InvalidQueryError("`alias_1` and `alias_2` must be distinct.")

        cols_1 = {c.lower() for c in meta1.column_metadata["column_name"]}
        cols_2 = {c.lower() for c in meta2.column_metadata["column_name"]}

        alias_kwargs = dict(
            collection_1_obj=collection_1_obj,
            catalog_1=catalog_1,
            collection_2_obj=collection_2_obj,
            catalog_2=catalog_2,
            alias_1=alias_1,
            alias_2=alias_2,
            cols_1=cols_1,
            cols_2=cols_2,
        )

        sort_by, search_criteria, columns, output_names, rename_blocks = self._prepare_crossmatch_columns(
            select_cols=select_cols,
            sort_by=sort_by,
            filters=filters,
            criteria=criteria,
            catalog_object_1=catalog_object_1,
            catalog_object_2=catalog_object_2,
            meta1=meta1,
            meta2=meta2,
            alias_kwargs=alias_kwargs,
        )

        # Prepare the ADQL region clause based on the provided region, object name, or coordinates
        adql_region = ""
        if region is not None:
            adql_region = self._create_adql_region(region)
        elif object_name is not None or coordinates is not None:
            coordinates = utils.parse_input_location(coordinates=coordinates, object_name=object_name,
                                                     resolver=resolver)
            adql_region = f"CIRCLE('ICRS', {coordinates.ra.deg}, {coordinates.dec.deg}, {search_radius_deg})"
        elif not search_criteria:
            warnings.warn(
                "No `region`, `coordinates`, `object_name`, or criteria were specified. This crossmatch will "
                "run against the entirety of both catalogs and may be slow or time out.",
                InputWarning,
            )

        ra1, dec1 = meta1.ra_column, meta1.dec_column
        ra2, dec2 = meta2.ra_column, meta2.dec_column
        distance = (
            f"DISTANCE(POINT('ICRS', {alias_1}.{ra1}, {alias_1}.{dec1}), "
            f"POINT('ICRS', {alias_2}.{ra2}, {alias_2}.{dec2}))"
        )
        if include_separation and not count_only:
            columns += f", {distance} * 3600 AS separation"

        # Prepare the SELECT clause, including the separation column if requested
        select_clause = "COUNT(*) AS count_all" if count_only else columns
        adql = f"SELECT TOP {1 if count_only else limit} {select_clause} "
        if adql_region:
            adql += (
                f"FROM (SELECT * FROM {catalog_1} WHERE "
                f"CONTAINS(POINT('ICRS', {ra1}, {dec1}), {adql_region}) = 1) AS {alias_1} "
            )
        else:
            adql += f"FROM {catalog_1} AS {alias_1} "
        adql += f"JOIN {catalog_2} AS {alias_2} "
        adql += f"ON {distance} < {match_radius_deg} "

        # Append any additional search criteria to the ADQL query
        if search_criteria:
            conditions = self._format_crossmatch_criteria(search_criteria, **alias_kwargs)
            if conditions:
                adql += "AND " + " AND ".join(conditions) + " "

        if sort_by and not count_only:
            sort_by = [sort_by] if isinstance(sort_by, str) else list(sort_by)
            sort_by = [
                "separation" if col == "separation" and include_separation and not count_only
                else self._qualify_crossmatch_column(col, allow_underscore=True, **alias_kwargs)
                for col in sort_by
            ]

            if isinstance(sort_desc, bool):
                sort_desc = [sort_desc]
            if len(sort_desc) not in (1, len(sort_by)):
                raise InvalidQueryError("Length of 'sort_desc' must be 1 or equal to length of 'sort_by'.")
            if len(sort_desc) == 1:
                sort_desc = sort_desc * len(sort_by)

            order_parts = [f"{col} {'DESC' if desc else 'ASC'}" for col, desc in zip(sort_by, sort_desc)]
            adql += "ORDER BY " + ", ".join(order_parts) + " "

        if offset and not count_only:
            adql += f"OFFSET {offset}"

        if return_adql:
            return adql.strip()

        # Run the ADQL query using the first collection's TAP service
        # Roman catalog is always used as the first collection in the crossmatch
        result_table = collection_1_obj.run_tap_query(adql, run_async=run_async)

        result_table.meta["adql_query"] = adql.strip()

        if len(result_table) == 0:
            warnings.warn("The query returned no results.", NoResultsWarning)

        # Return the count if requested, otherwise return the full result table with metadata
        if count_only:
            return int(result_table["count_all"][0])
        else:
            result_table.meta["collection1"] = collection_1_obj.name
            result_table.meta["catalog1"] = catalog_1
            result_table.meta["collection2"] = collection_2_obj.name
            result_table.meta["catalog2"] = catalog_2
            result_table.meta["alias1"] = alias_1
            result_table.meta["alias2"] = alias_2
            self._relabel_crossmatch_columns(result_table, output_names, rename_blocks, include_separation)
            if include_separation and "separation" in result_table.colnames:
                result_table["separation"].unit = u.arcsec
        return result_table

    @class_or_instance
    @deprecated(since="v0.4.13", message=("This function is deprecated and will be removed in a future release."))
    def query_hsc_matchid_async(self, match, *, version=3, pagesize=None, page=None):
        """
        Returns all the matches for a given Hubble Source Catalog MatchID.

        Parameters
        ----------
        match : int or `~astropy.table.Row`
            The matchID or HSC entry to return matches for.
        version : int, optional
            The HSC version to match against. Default is v3.
        pagesize : int, optional
            Can be used to override the default pagesize.
            E.g. when using a slow internet connection.
        page : int, optional
            Can be used to override the default behavior of all results being returned to obtain
            one specific page of results.

        Returns
        -------
        response : list of `~requests.Response`
        """
        self._current_connection = self._portal_api_connection

        if isinstance(match, Row):
            match = match["MatchID"] if "MatchID" in match.colnames else match["matchid"]
        match = str(match)  # np.int64 gives json serializer problems, so stringify right here

        if version == 2:
            service = "Mast.HscMatches.Db.v2"
        else:
            if version not in (3, None):
                warnings.warn("Invalid HSC version number, defaulting to v3.", InputWarning)
            service = "Mast.HscMatches.Db.v3"

        params = {"input": match}

        return self._current_connection.service_request_async(service, params, pagesize, page)

    @class_or_instance
    @deprecated(since="v0.4.13", message=("This function is deprecated and will be removed in a future release."))
    def get_hsc_spectra_async(self, *, pagesize=None, page=None):
        """
        Returns all Hubble Source Catalog spectra.

        Parameters
        ----------
        pagesize : int, optional
            Can be used to override the default pagesize.
            E.g. when using a slow internet connection.
        page : int, optional
            Can be used to override the default behavior of all results being returned to obtain
            one specific page of results.

        Returns
        -------
        response : list of `~requests.Response`
        """
        self._current_connection = self._portal_api_connection
        return self._current_connection.service_request_async("Mast.HscSpectra.Db.All", {}, pagesize, page)

    @deprecated(since="v0.4.13", message=("This function is deprecated and will be removed in a future release."))
    def download_hsc_spectra(self, spectra, *, download_dir=None, cache=True, curl_flag=False):
        """
        Download one or more Hubble Source Catalog spectra.

        Parameters
        ----------
        spectra : `~astropy.table.Table` or `~astropy.table.Row`
            One or more HSC spectra to be downloaded.
        download_dir : str, optional
           Specify the base directory to download spectra into.
           Spectra will be saved in the subdirectory download_dir/mastDownload/HSC.
           If download_dir is not specified the base directory will be '.'.
        cache : bool, optional
            Default is True. If file is found on disc it will not be downloaded again.
            Note: has no affect when downloading curl script.
        curl_flag : bool, optional
            Default is False.  If true instead of downloading files directly, a curl script
            will be downloaded that can be used to download the data files at a later time.

        Returns
        -------
        response : list of `~requests.Response`
        """
        # Normalize spectra input to a list
        if isinstance(spectra, Row):
            spectra = [spectra]

        # Ensure download directory is set
        download_dir = download_dir or "."

        if curl_flag:
            timestamp = time.strftime("%Y%m%d%H%M%S")
            bundle_name = "mastDownload_" + timestamp
            url_list = [self._make_data_url(spec) for spec in spectra]
            path_list = [f"{bundle_name}/HSC/{spec['DatasetName']}.fits" for spec in spectra]

            params = dict(
                urlList=",".join(url_list),
                filename=bundle_name,
                pathList=",".join(path_list),
                descriptionList=[""] * len(spectra),
                productTypeList=["spectrum"] * len(spectra),
                extension="curl",
            )

            service = "Mast.Bundle.Request"
            response = self._portal_api_connection.service_request_async(service, params)
            bundle_info = response[0].json()
            local_script = os.path.join(download_dir, f"{bundle_name}.sh")
            self._download_file(bundle_info["url"], local_script, head_safe=True)

            # Build manifest row
            exists = os.path.isfile(local_script)
            missing = [k for k, v in bundle_info.get("statusList", {}).items() if v != "COMPLETE"]
            manifest = Table(
                {
                    "Local Path": [local_script],
                    "Status": ["COMPLETE" if exists else "ERROR"],
                    "Message": [
                        None
                        if exists and not missing
                        else (
                            f"{len(missing)} files could not be added to curl script"
                            if exists
                            else "Curl script could not be downloaded"
                        )
                    ],
                    "URL": [None if exists and not missing else (",".join(missing) if missing else bundle_info["url"])],
                }
            )
            return manifest

        base_dir = os.path.join(download_dir, "mastDownload", "HSC")
        os.makedirs(base_dir, exist_ok=True)
        manifest_rows = []

        for row in spectra:
            dataset = row["DatasetName"]
            url = self._make_data_url(row)
            local_path = os.path.join(base_dir, f"{dataset}.fits")
            status = "COMPLETE"
            message = None

            try:
                self._download_file(url, local_path, cache=cache, head_safe=True)

                if not os.path.exists(local_path):
                    status = "ERROR"
                    message = "File was not downloaded"
            except requests.HTTPError as err:
                status = "ERROR"
                message = f"HTTPError: {err}"

            manifest_rows.append([local_path, status, message, url])

        return Table(rows=manifest_rows, names=("Local Path", "Status", "Message", "URL"))

    def _parse_result(self, response, *, verbose=False):
        """Parse the async responses from HSC queries."""
        return self._current_connection._parse_result(response, verbose=verbose)

    def _verify_collection(self, collection):
        """
        Verify that the specified collection is valid and return the correct collection name.
        Warns the user if the collection has been renamed and raises an error if the collection is not valid.

        Parameters
        ----------
        collection : str
            The collection to be verified.

        Raises
        ------
        InvalidQueryError
            If the specified collection is not valid.
        """
        collection = collection.lower().strip()
        if collection in self.available_collections:
            return collection
        else:
            if collection in self._renamed_collections:
                new_name = self._renamed_collections[collection]
                warn_msg = f"Collection '{collection}' has been renamed. Please use '{new_name}' instead."
                warnings.warn(warn_msg, InputWarning)
                return new_name

            error_msg = ""
            if collection in self._no_longer_supported_collections:
                error_msg = (
                    f"Collection '{collection}' is no longer supported. To query from this catalog, "
                    f"please use a version of Astroquery older than 0.4.13. You can install version 0.4.12 using "
                    f"`pip install astroquery==0.4.12`"
                )
            else:
                closest = difflib.get_close_matches(collection, self.available_collections, n=1)
                suggestion = f" Did you mean '{closest[0]}'?" if closest else ""
                error_msg = f"Collection '{collection}' is not recognized.{suggestion}"
            error_msg += " Available collections are: " + ", ".join(self.available_collections)
            raise InvalidQueryError(error_msg)

    def _get_collection_obj(self, collection_name):
        """
        Given a collection name, find or create the corresponding CatalogCollection object.

        Parameters
        ----------
        collection_name : str
            The name of the collection.

        Returns
        -------
        CatalogCollection
            The corresponding CatalogCollection object.
        """
        if not isinstance(collection_name, str):
            raise InvalidQueryError("Collection name must be a string.")

        collection_name = collection_name.lower().strip()
        if collection_name in self._collections_cache:
            return self._collections_cache[collection_name]

        collection_name = self._verify_collection(collection_name)
        collection_obj = CatalogCollection(collection_name)
        self._collections_cache[collection_name] = collection_obj
        return collection_obj

    def _parse_inputs(self, collection=None, catalog=None):
        """
        Parse and validate the collection and catalog inputs.

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
        collection_obj = self._get_collection_obj(collection) if collection else self._collection

        if not catalog:
            # If the class attribute catalog is valid for the collection, use it
            # Otherwise, use the default catalog for the collection
            if self.catalog in collection_obj.catalog_names:
                catalog = self.catalog
            else:
                catalog = collection_obj.default_catalog
        else:
            catalog = catalog.lower()
            # For backwards compatibility, check if the user is trying to specify a collection via catalog
            if (
                catalog in self.available_collections
                or catalog in self._no_longer_supported_collections
                or catalog in self._renamed_collections
            ) and not collection:
                warnings.warn(
                    f"Specifying collection '{catalog}' via the `catalog` parameter is deprecated. "
                    f"Please use the `collection` parameter instead.",
                    DeprecationWarning,
                )
                # As a convenience to the user, set the collection accordingly and use its default catalog
                collection_obj = self._get_collection_obj(catalog)
                catalog = collection_obj.default_catalog
            else:
                catalog = collection_obj._verify_catalog(catalog)

        return collection_obj, catalog

    def _parse_select_cols(self, select_cols, column_metadata):
        """
        Validate and parse the select_cols parameter.

        Parameters
        ----------
        select_cols : list of str
            List of column names to include in the result.
        column_metadata : `~astropy.table.Table`
            The catalog's column metadata table.

        Returns
        -------
        str
            Comma-separated string of valid column names for ADQL SELECT clause.

        Raises
        ------
        InvalidQueryError
            If any specified column is not found in the catalog metadata.
        """
        valid_columns = column_metadata["column_name"].tolist()
        valid_selected = []
        for col in select_cols:
            if col not in valid_columns:
                closest = difflib.get_close_matches(col, valid_columns, n=1)
                suggestion = f" Did you mean '{closest[0]}'?" if closest else ""
                warnings.warn(f"Column '{col}' not found in catalog.{suggestion}", InputWarning)
            else:
                valid_selected.append(col)
        if not valid_selected:
            raise InvalidQueryError("No valid columns specified in `select_cols`.")
        return ", ".join(valid_selected)

    def _parse_crossmatch_target(self, target):
        """
        Resolve a compact crossmatch target without changing instance state.

        Parameters
        ----------
        target : str or tuple of str
            The target to resolve, either as a collection name string or a (collection, catalog) pair.

        Returns
        -------
        tuple of str
            A tuple containing the resolved collection name and catalog name.

        Raises
        ------
        InvalidQueryError
            If the target is not a valid collection name or (collection, catalog) pair.
        """
        if isinstance(target, str) and target.strip():
            target = target.strip()
            collection_obj = self._get_collection_obj(target)
            return collection_obj.name, collection_obj.default_catalog

        if (isinstance(target, (tuple, list)) and len(target) == 2
                and all(isinstance(value, str) and value.strip() for value in target)):
            return tuple(value.strip() for value in target)

        raise InvalidQueryError("Each target must be a collection name or a (collection, catalog) pair of strings.")

    def _resolve_crossmatch_targets(self, collection_1, catalog_1, collection_2, catalog_2, catalogs):
        """
        Validate and normalize the two target specifications for ``crossmatch``.

        Parameters
        ----------
        collection_1 : str or None
            The name of the first collection, or None if using `catalogs`.
        catalog_1 : str or None
            The name of the first catalog, or None if using `catalogs`.
        collection_2 : str or None
            The name of the second collection, or None if using `catalogs`.
        catalog_2 : str or None
            The name of the second catalog, or None if using `catalogs`.
        catalogs : list or tuple of two elements, optional
            A list or tuple containing exactly two Catalog objects, collections, or (collection, catalog) pairs.

        Returns
        -------
        tuple of tuple of str
            A tuple containing two tuples, each with the resolved collection and catalog names.

        Raises
        ------
        InvalidQueryError
            If the input specifications are invalid or inconsistent.
        """
        from .catalog import Catalog

        if catalogs is not None:
            if any(value is not None for value in (collection_1, catalog_1, collection_2, catalog_2)):
                raise InvalidQueryError(
                    "Specify either `catalogs` or the collection/catalog arguments, not both."
                )

            if not isinstance(catalogs, (list, tuple)) or len(catalogs) != 2:
                raise InvalidQueryError(
                    "`catalogs` must contain exactly two Catalog objects, collections, "
                    "or (collection, catalog) pairs."
                )

            target_inputs = tuple(
                (None, target) if isinstance(target, Catalog) else self._parse_crossmatch_target(target)
                for target in catalogs
            )
        elif all(value is None for value in (collection_1, catalog_1, collection_2, catalog_2)):
            raise InvalidQueryError(
                "Specify two catalogs to crossmatch, for example with `catalogs=(collection_1, collection_2)`."
            )
        else:
            # Use the individual collection and catalog arguments to construct target_inputs
            target_inputs = ((collection_1, catalog_1), (collection_2, catalog_2))

        def resolve_target(collection, catalog, suffix):
            """Resolve a crossmatch target to its collection and catalog names."""
            collection_object = collection if isinstance(collection, Catalog) else None
            catalog_object = catalog if isinstance(catalog, Catalog) else None

            if collection_object is not None and catalog_object is not None:
                if collection_object is not catalog_object:
                    raise InvalidQueryError(
                        f"`collection_{suffix}` and `catalog_{suffix}` specify different Catalog objects."
                    )
            elif collection_object is not None and catalog is not None:
                raise InvalidQueryError(
                    f"Do not specify `catalog_{suffix}` when `collection_{suffix}` is a Catalog object."
                )

            target = catalog_object or collection_object
            if target is None:
                return collection, catalog, None

            if collection is not None and not isinstance(collection, Catalog) and (
                not isinstance(collection, str)
                or collection.strip().casefold() != target.collection_name.casefold()
            ):
                raise InvalidQueryError(
                    f"`collection_{suffix}` conflicts with the Catalog object in `catalog_{suffix}`."
                )

            return target.collection_name, target.catalog_name, target

        return tuple(
            resolve_target(collection, catalog, str(index))
            for index, (collection, catalog) in enumerate(target_inputs, start=1)
        )

    def _normalize_columns(self, columns, resolve_column, column_reference_type, *, scalar_string_as_list=False):
        """Normalize nested column groups while resolving catalog-owned references."""
        if isinstance(columns, column_reference_type):
            return [resolve_column(columns)]
        if isinstance(columns, str):
            return [resolve_column(columns)] if scalar_string_as_list else columns

        normalized = []
        for column in columns:
            if isinstance(column, (tuple, list)):
                normalized.extend(self._normalize_columns(
                    column, resolve_column, column_reference_type,
                    scalar_string_as_list=scalar_string_as_list,
                ))
            else:
                normalized.append(resolve_column(column))
        return normalized

    def _prepare_crossmatch_columns(
        self, *, select_cols, sort_by, filters, criteria, catalog_object_1, catalog_object_2,
        meta1, meta2, alias_kwargs,
    ):
        """
        Normalize column references and prepare criteria and output columns for ``crossmatch``.

        Parameters
        ----------
        select_cols : list or str or None
            Columns to select in the crossmatch query.
        sort_by : list or str or None
            Columns to sort the results by.
        filters : dict or None
            Mapping of columns to filter criteria.
        criteria : list of columns
            Columns used for crossmatch criteria.
        catalog_object_1 : Catalog or None
            The first catalog object involved in the crossmatch.
        catalog_object_2 : Catalog or None
            The second catalog object involved in the crossmatch.
        meta1 : dict
            Metadata for the first catalog.
        meta2 : dict
            Metadata for the second catalog.
        alias_kwargs : dict
            Additional keyword arguments for alias resolution.

        Returns
        -------
        tuple
            Normalized select columns, sort columns, filters, and criteria columns.

        Raises
        ------
        InvalidQueryError
            If column references are invalid or inconsistent.
        """
        from .catalog import CatalogColumn

        def qualify_catalog_column(column):
            """Qualify a catalog column with its catalog prefix for crossmatch queries."""
            if not isinstance(column, CatalogColumn):
                return column
            if catalog_object_1 is not None and column.catalog is catalog_object_1:
                return f"cat1.{column.name}"
            if catalog_object_2 is not None and column.catalog is catalog_object_2:
                return f"cat2.{column.name}"
            raise InvalidQueryError("Column references must belong to one of the crossmatch Catalog objects.")

        # Normalize select and sort columns
        if select_cols is not None:
            if isinstance(select_cols, str):
                select_cols = [column.strip() for column in select_cols.split(",")]
            select_cols = self._normalize_columns(select_cols, qualify_catalog_column, CatalogColumn)
        if sort_by is not None:
            sort_by = self._normalize_columns(sort_by, qualify_catalog_column, CatalogColumn)

        # Normalize and validate filter columns
        normalized_filters = {}
        filter_columns = set()
        if filters is not None:
            if not isinstance(filters, Mapping):
                raise InvalidQueryError("`filters` must be a mapping of columns to criteria.")
            for column, value in filters.items():
                column = qualify_catalog_column(column)
                resolved = self._qualify_crossmatch_column(column, allow_underscore=True, **alias_kwargs)
                if resolved.casefold() in filter_columns:
                    raise InvalidQueryError(f"Criteria column '{resolved}' was specified more than once.")
                filter_columns.add(resolved.casefold())
                normalized_filters[column] = value

        # Check for overlap between criteria and filter columns
        criteria_columns = {
            self._qualify_crossmatch_column(column, allow_underscore=True, **alias_kwargs).casefold()
            for column in criteria
        }
        overlap = criteria_columns & filter_columns
        if overlap:
            raise InvalidQueryError(
                f"Criteria specified both as keyword arguments and in 'filters' for columns: "
                f"{', '.join(sorted(overlap))}; duplicate columns were specified more than once."
            )

        # Combine criteria and normalized filters into search criteria
        search_criteria = {**criteria, **normalized_filters}

        # Parse and prepare select columns for the crossmatch query
        if select_cols:
            columns, output_names = self._parse_crossmatch_select_cols(select_cols, **alias_kwargs)
            rename_blocks = None
        else:
            # If no select columns are specified, default to selecting all columns from both catalogs
            columns = f"{alias_kwargs['alias_1']}.*, {alias_kwargs['alias_2']}.*"
            output_names = None
            rename_blocks = [
                (alias_kwargs["alias_1"], list(meta1.column_metadata["column_name"])),
                (alias_kwargs["alias_2"], list(meta2.column_metadata["column_name"])),
            ]

        return sort_by, search_criteria, columns, output_names, rename_blocks

    def _parse_crossmatch_radius(self, value, name):
        """
        Convert a positive, finite scalar angular radius to degrees.

        Parameters
        ----------
        value : float or astropy.units.Quantity
            The angular radius to convert.
        name : str
            The name of the parameter for error messages.

        Returns
        -------
        float
            The angular radius in degrees.

        Raises
        ------
        InvalidQueryError
            If the value is not a finite positive scalar angle or exceeds the parameter-specific maximum:
            5 arcseconds for ``match_radius`` or 30 arcminutes for ``search_radius``.
        """
        maximum = {"match_radius": 5 * u.arcsec, "search_radius": 30 * u.arcmin}[name]
        maximum_degrees = maximum.to_value(u.deg)
        try:
            angle = coord.Angle(value, u.deg)
            if angle.isscalar:
                degrees = float(angle.to_value(u.deg))
                if math.isfinite(degrees) and 0 < degrees <= maximum_degrees:
                    return degrees
        except (TypeError, ValueError, u.UnitsError):
            pass
        raise InvalidQueryError(f"`{name}` must be a finite scalar angle greater than 0 and at most {maximum}.")

    def _qualify_crossmatch_column(self, col, *, alias_1, alias_2, cols_1, cols_2, allow_underscore=False, **kwargs):
        """
        Qualify a column name referenced in a `crossmatch` query with the appropriate table alias.

        Parameters
        ----------
        col : str
            The column name, optionally already qualified as ``'alias.column'``.
        alias_1 : str
            The ADQL table alias for the first catalog.
        alias_2 : str
            The ADQL table alias for the second catalog.
        cols_1 : set of str
            Lowercased column names belonging to the first catalog.
        cols_2 : set of str
            Lowercased column names belonging to the second catalog.
        allow_underscore : bool, optional
            Accept ``alias_column`` shorthand. Real column names take precedence.

        Returns
        -------
        str
            The column name qualified with its owning table alias (e.g. ``'gaiadr3.ra'``).
        """
        if not isinstance(col, str) or not col.strip():
            raise InvalidQueryError("Crossmatch column names must be non-empty strings.")

        col = col.strip()
        aliases = {
            alias_1.casefold(): alias_1,
            alias_2.casefold(): alias_2,
            "cat1": alias_1,
            "cat2": alias_2,
        }

        if allow_underscore and "." not in col and col.lower() not in cols_1 and col.lower() not in cols_2:
            # Match complete aliases, longest first: aliases and column names may contain underscores
            for prefix in sorted(aliases, key=len, reverse=True):
                if col.casefold().startswith(prefix + "_"):
                    col = f"{aliases[prefix]}.{col[len(prefix) + 1:]}"
                    break

        if "." in col:
            # If the column is already qualified with a table alias, validate it
            alias, name = (part.strip() for part in col.split(".", 1))
            resolved_alias = aliases.get(alias.casefold())
            if resolved_alias is None:
                raise InvalidQueryError(
                    f"Unrecognized table alias '{alias}' in column '{col}'. "
                    f"Use '{alias_1}', '{alias_2}', 'cat1', or 'cat2'."
                )
            valid_columns = cols_1 if resolved_alias == alias_1 else cols_2
            if name.lower() not in valid_columns:
                closest = difflib.get_close_matches(name.lower(), sorted(valid_columns), n=1)
                suggestion = f" Did you mean '{resolved_alias}.{closest[0]}'?" if closest else ""
                raise InvalidQueryError(f"Column '{col}' was not found in '{resolved_alias}'.{suggestion}")
            return f"{resolved_alias}.{name}"

        # If the column is not qualified with a table alias, determine which catalog it belongs to
        lc = col.lower()
        in_1 = lc in cols_1
        in_2 = lc in cols_2
        if in_1 and in_2:
            raise InvalidQueryError(
                f"Column '{col}' exists in both catalogs and must be qualified with a table alias, "
                f"e.g. '{alias_1}.{col}' or '{alias_2}.{col}'."
            )
        if in_1:
            return f"{alias_1}.{col}"
        if in_2:
            return f"{alias_2}.{col}"
        raise InvalidQueryError(f"Column '{col}' was not found in either '{alias_1}' or '{alias_2}'.")

    def _parse_crossmatch_select_cols(self, select_cols, **alias_kwargs):
        """
        Validate and qualify the select_cols parameter for a `crossmatch` query.

        Parameters
        ----------
        select_cols : list of str
            List of column names to include in the result.
        **alias_kwargs
            Keyword arguments accepted by `_qualify_crossmatch_column`.

        Returns
        -------
        columns : str
            Comma-separated string of alias-qualified column names for the ADQL SELECT clause.
        output_names : list of str
            The ``'<alias>.<column>'`` names, in SELECT order, that the result columns
            will be renamed to after the query runs.
        """
        output_names = []
        seen = set()
        for col in select_cols:
            qualified = self._qualify_crossmatch_column(col, allow_underscore=True, **alias_kwargs)
            if qualified.casefold() in seen:
                raise InvalidQueryError(f"Duplicate output column '{qualified}'. Choose distinct columns or aliases.")
            seen.add(qualified.casefold())
            output_names.append(qualified)
        return ", ".join(output_names), output_names

    @staticmethod
    def _relabel_crossmatch_columns(result_table, output_names, rename_blocks, include_separation):
        """
        Rename the columns of a `crossmatch` result table to the ``'<alias>.<column>'`` form.

        Output aliases are omitted from the generated ADQL to keep the query short (and because
        ADQL output aliases cannot contain dots), so the server labels result columns itself.
        Columns are therefore relabeled positionally, since a TAP result preserves the order of
        the SELECT clause.

        Parameters
        ----------
        result_table : `~astropy.table.Table`
            The crossmatch result table, modified in place.
        output_names : list of str or None
            Desired ``'<alias>.<column>'`` names in SELECT order when explicit columns were
            selected, or None when ``alias.*`` was used for both catalogs.
        rename_blocks : list of tuple or None
            For the ``alias.*`` case, ``(alias, column_names)`` pairs per catalog, in SELECT
            order, with column names in catalog metadata order.
        include_separation : bool
            Whether the query appended a ``separation`` column after the selected columns.

        Notes
        -----
        The function assumes that the order of columns in the result table matches the order of the
        SELECT clause in the query.
        """
        returned = list(result_table.colnames)
        expected = len(output_names) if output_names is not None else sum(len(n) for _, n in rename_blocks)

        if len(returned) != expected + (1 if include_separation else 0):
            warnings.warn(
                "The query returned an unexpected number of columns, so result columns could not "
                "be relabeled to the '<alias>.<column>' form.", AstropyWarning,
            )
            return

        if output_names is None:
            output_names = []
            index = 0
            # Iterate over each catalog and its columns to build the output names list
            for alias, names in rename_blocks:
                valid = {name.casefold() for name in names}
                for expected_name in names:
                    actual = returned[index]
                    # Prefer the server-reported name when it is a real column of the catalog;
                    # names mangled to stay unique fall back to the catalog's column order.
                    name = actual if actual.casefold() in valid else expected_name
                    output_names.append(f"{alias}.{name}")
                    index += 1
        for old, new in zip(returned[:expected], output_names):
            if old != new:
                result_table.rename_column(old, new)

    def _format_crossmatch_criteria(
        self, criteria, *, collection_1_obj, catalog_1, collection_2_obj, catalog_2, alias_1, alias_2,
        cols_1, cols_2,
    ):
        """
        Turn a `crossmatch` criteria dict into alias-qualified ADQL WHERE clause expressions.

        Parameters
        ----------
        criteria : dict
            Mapping of column name (optionally alias-qualified) to scalar or list of scalars.
        collection_1_obj, collection_2_obj : CatalogCollection
            The collection objects for the first and second catalogs.
        catalog_1, catalog_2 : str
            The catalog names for the first and second catalogs.
        alias_1, alias_2 : str
            The ADQL table aliases for the first and second catalogs.
        cols_1, cols_2 : set of str
            Lowercased column names belonging to each catalog.

        Returns
        -------
        list of str
            ADQL predicate strings, alias-qualified and suitable for joining with ' AND '.
        """
        conditions = []
        seen_columns = {}

        # Iterate over each criterion and convert it into an ADQL condition
        for raw_key, value in criteria.items():
            qualified = self._qualify_crossmatch_column(
                raw_key, alias_1=alias_1, alias_2=alias_2, cols_1=cols_1, cols_2=cols_2, allow_underscore=True,
            )
            normalized = qualified.casefold()
            if normalized in seen_columns:
                raise InvalidQueryError(
                    f"Criteria column '{qualified}' was specified more than once as "
                    f"'{seen_columns[normalized]}' and '{raw_key}'."
                )
            seen_columns[normalized] = raw_key
            alias, key = qualified.split(".", 1)
            target_obj, target_catalog = (
                (collection_1_obj, catalog_1) if alias == alias_1 else (collection_2_obj, catalog_2)
            )
            conditions.extend(self._format_criteria_conditions(
                target_obj, target_catalog, {key.lower(): value}, column_alias=alias,
            ))
        return conditions

    def _parse_legacy_pagination(self, limit, offset, pagesize, page):
        """
        Parse legacy pagesize and page parameters to determine limit and offset.

        Parameters
        ----------
        limit : int
            The maximum number of results to return.
        offset : int
            The number of rows to skip before starting to return rows.
        pagesize : int, optional
            The number of results per page (legacy parameter).
        page : int, optional
            The page number to return (legacy parameter).

        Returns
        -------
        tuple
            A tuple containing the (limit, offset) values.
        """
        # If limit and offset are default, check for legacy pagination params
        if limit == 5000 and offset == 0:
            if pagesize is not None:
                if page is None:
                    page = 1  # Default to first page if not specified
                limit = pagesize
                offset = (page - 1) * pagesize
            elif page is not None:
                warnings.warn(
                    "The 'page' parameter is ignored without 'pagesize'. "
                    "Please use `limit` and `offset` to specify pagination.",
                    InputWarning,
                )
        return limit, offset

    def _create_adql_region(self, region):
        """
        Returns the ADQL description of the given polygon or circle region.

        Parameters
        ----------
        region : str | iterable | astropy.regions.Region
            - Iterable of RA/Dec pairs as lists/sequences
            - STC-S POLYGON or CIRCLE string
            - `~astropy.regions.CircleSkyRegion` or `~astropy.regions.PolygonSkyRegion`

        Returns
        -------
        adql_region : str
            ADQL representation of the region (POLYGON or CIRCLE)
        """
        # Case 1: region is a string (e.g. STC-S syntax)
        if isinstance(region, str):
            parts = region.strip().lower().split()
            shape = parts[0]

            if shape == "polygon":
                # POLYGON lon1 lat1 lon2 lat2 ...
                # Optional format: POLYGON ICRS lon1 lat1 ...
                # parts = ["POLYGON", maybe_frame?, ...coords...]

                # Determine if parts[1] is a coord or a frame name
                if len(parts) < 3:
                    raise InvalidQueryError(f"Invalid POLYGON region string: {region}")
                try:
                    float(parts[1])  # numeric → no frame name
                    point_parts = parts[1:]
                except ValueError:
                    point_parts = parts[2:]  # skip optional frame name

                if len(point_parts) < 6 or len(point_parts) % 2 != 0:
                    # polygon requires at least 3 points (6 numbers), and must be pairs
                    raise InvalidQueryError(f"Invalid POLYGON region string: {region}")

                point_string = ",".join(point_parts)
                return f"POLYGON('ICRS',{point_string})"

            elif shape == "circle":
                # CIRCLE ra dec radius   (or CIRCLE ICRS ra dec radius)
                if len(parts) < 4:
                    raise InvalidQueryError(f"Invalid CIRCLE region string: {region}")

                # Try interpreting parts[1] as RA. If not numeric, assume it's a frame
                try:
                    float(parts[1])
                    # Format: CIRCLE ra dec radius
                    ra, dec, radius = parts[1], parts[2], parts[3]
                except ValueError:
                    # Format: CIRCLE FRAME ra dec radius
                    if len(parts) < 5:
                        raise InvalidQueryError(f"Invalid CIRCLE region string: {region}")
                    ra, dec, radius = parts[2], parts[3], parts[4]

                return f"CIRCLE('ICRS',{ra},{dec},{radius})"

            else:
                raise InvalidQueryError(f"Unrecognized region string: {region}")

        # Case 2: region is an astropy region object
        # TODO: When released, change these to use `CircleSphericalSkyRegion` and `PolygonSphericalSkyRegion`
        if HAS_REGIONS:
            if isinstance(region, CircleSkyRegion):
                center = region.center.icrs
                radius = region.radius.to(u.deg).value
                return f"CIRCLE('ICRS',{center.ra.deg},{center.dec.deg},{radius})"
            elif isinstance(region, PolygonSkyRegion):
                verts = region.vertices.icrs
                point_string = ",".join(f"{v.ra.deg},{v.dec.deg}" for v in verts)
                return f"POLYGON('ICRS',{point_string})"

        # Case 3: region is an iterable of coordinate pairs
        if isinstance(region, Iterable):
            # Expect something like [(ra1, dec1), (ra2, dec2), ...]
            try:
                points = [float(x) for point in region for x in point]
            except Exception as e:
                raise InvalidQueryError(f"Invalid iterable region format: {region}") from e
            return f"POLYGON('ICRS',{','.join(str(x) for x in points)})"

        else:
            raise TypeError(f"Unsupported region type: {type(region)}")

    def _classify_columns(self, meta):
        """
        Classify columns as numeric or temporal based on their datatype and UCD metadata.

        Parameters
        ----------
        meta : `~astropy.table.Table`
            The catalog's column metadata table, which must include 'column_name', 'datatype', and 'ucd' columns.

        Returns
        -------
        tuple
            A tuple containing two sets: (numeric_columns, temporal_columns), where each set contains the names of the
            columns classified as numeric or temporal, respectively.
        """
        num_types = {"int", "long", "short", "float", "double", "floatcomplex", "doublecomplex"}
        temporal_tokens = {"date", "time", "timestamp", "datetime"}

        numeric = set()
        temporal = set()

        for name, dtype, ucd in zip(meta["column_name"], meta["datatype"], meta["ucd"]):
            dtype_str = dtype.lower() if isinstance(dtype, str) else ""
            ucd_str = ucd.lower() if isinstance(ucd, str) else ""

            # Classify as numeric if the datatype matches known numeric types
            if dtype_str in num_types:
                numeric.add(name)

            # Classify as temporal if the datatype contains temporal tokens or if the
            # UCD indicates a time-related quantity (but exclude numeric types that may have time-related UCDs)
            has_temporal_dtype = any(token in dtype_str for token in temporal_tokens)
            has_temporal_ucd = ucd_str.startswith("time.") or ";time." in ucd_str
            not_numeric = dtype_str not in num_types

            if has_temporal_dtype or (has_temporal_ucd and not_numeric):
                temporal.add(name)

        return numeric, temporal

    def _quote_adql_string(self, adql_str):
        """Escape single quotes in ADQL query strings by doubling them."""
        return adql_str.replace("'", "''")

    def _parse_range_cmp_expr(self, col, expr, value_formatter):
        expr = expr.strip()

        range_match = re.fullmatch(r"(.+?)\s*\.\.\s*(.+)", expr)
        if range_match:
            low = value_formatter(range_match.group(1).strip(), col)
            high = value_formatter(range_match.group(2).strip(), col)
            return f"{col} BETWEEN {low} AND {high}"

        cmp_match = re.fullmatch(r"(<=|>=|<|>)\s*(.+)", expr)
        if cmp_match:
            rhs = value_formatter(cmp_match.group(2).strip(), col)
            return f"{col} {cmp_match.group(1)} {rhs}"

    def _parse_numeric_expr(self, col, expr):
        """
        Parse a numeric expression for a column and return the corresponding ADQL predicate.

        Parameters
        ----------
        col : str
            The column name.
        expr : str
            The numeric expression (e.g., "5", "<10", "5..10").

        Returns
        -------
        str
            The ADQL predicate for the numeric expression.
        """
        parsed = self._parse_range_cmp_expr(col, expr, lambda x, _: x)
        if parsed:
            return parsed

        try:
            return f"{col} = {float(expr)}"
        except ValueError:
            raise InvalidQueryError(
                f"Column '{col}' is numeric; unsupported value '{expr}'. Use numbers, comparisons like '<10', or "
                "ranges like '5..10'."
            )

    def _normalize_time(self, value):
        """
        Normalize a datetime value to a string format suitable for ADQL queries.

        Parameters
        ----------
        value : str or `~astropy.time.Time` or `~datetime.datetime`
            The datetime value to normalize.

        Returns
        -------
        str
            The normalized datetime string in the format 'YYYY-MM-DD HH:MM:SS' or the original
            value if it cannot be parsed as a datetime.

        Raises
        ------
        ValueError
            If the value cannot be parsed as a datetime.
        """
        t = Time(value)
        dt = t.to_datetime()

        # Drop microseconds to avoid ADQL parsing issues
        if dt.microsecond:
            dt = dt.replace(microsecond=0)

        return dt.strftime("%Y-%m-%d %H:%M:%S")

    def _to_temporal_literal(self, value, col):
        """
        Convert a datetime value to an ADQL literal string, ensuring it is properly formatted and quoted.

        Parameters
        ----------
        value : str or `~astropy.time.Time` or `~datetime.datetime`
            The datetime value to convert.
        col : str
            The column name (used for error messages).

        Returns
        -------
        str
            The ADQL literal string representing the datetime value.
        """
        try:
            normalized = self._normalize_time(value)
        except ValueError:
            # If it can't be parsed as a time, send the value as-is for the backend to handle
            # (which may raise an error if it's invalid)
            normalized = value if isinstance(value, str) else str(value)
        escaped = self._quote_adql_string(normalized)
        return f"'{escaped}'"

    def _parse_temporal_expr(self, col, expr):
        """
        Parse a datetime expression for a column and return the corresponding ADQL predicate.

        Parameters
        ----------
        col : str
            The column name.
        expr : str
            Datetime expression (e.g., "2024-01-01", ">=2024-01-01", "2024-01-01..2024-12-31").

        Returns
        -------
        str
            The ADQL predicate for the datetime expression.
        """
        if isinstance(expr, str):
            parsed = self._parse_range_cmp_expr(col, expr, self._to_temporal_literal)
            if parsed:
                return parsed

        # For simple equality, expand to a small range
        try:
            t0 = Time(self._normalize_time(expr))
            t1 = t0 + 1 * u.s
            low = self._quote_adql_string(t0.strftime("%Y-%m-%d %H:%M:%S"))
            high = self._quote_adql_string(t1.strftime("%Y-%m-%d %H:%M:%S"))
            return f"{col} BETWEEN '{low}' AND '{high}'"
        except ValueError:
            # If it can't be parsed as a time, send the value as-is for the backend to handle
            # (which may raise an error if it's invalid)
            return f"{col} = '{expr}'"

    def _format_scalar_predicate(self, col, val, is_numeric=False, is_temporal=False):
        """
        Build predicate for a scalar value, aware of column type.

        Parameters
        ----------
        col : str
            The column name.
        val : scalar
            The value to build the predicate for.
        is_numeric : bool
            Whether the column is numeric.
        is_temporal : bool
            Whether the column is temporal.

        Returns
        -------
        str
            The ADQL predicate for the scalar value.
        """
        if isinstance(val, bool):
            # Booleans stored as integers
            return f"{col} = {int(val)}"

        if is_temporal:
            is_neg = isinstance(val, str) and val.startswith("!")
            sval = val[1:].strip() if is_neg else val
            parsed = self._parse_temporal_expr(col, sval)
            return f"NOT ({parsed})" if is_neg else parsed

        if isinstance(val, str):
            # Check for negation
            is_neg = val.startswith("!")
            sval = val[1:].strip() if is_neg else val

            # Strings for numeric columns
            if is_numeric:
                parsed = self._parse_numeric_expr(col, sval)
                return f"NOT ({parsed})" if is_neg else parsed

            # Non-numeric strings
            has_wild = ("*" in sval) or ("%" in sval)
            pattern = self._quote_adql_string(sval.replace("*", "%"))
            expr = f"{col} LIKE '{pattern}'" if has_wild else f"{col} = '{pattern}'"
            return f"NOT ({expr})" if is_neg else expr

        # Numerics or others
        return f"{col} = {val}"

    def _combine_predicates(self, pos_parts, neg_parts):
        """
        Combine positive and negative predicate parts into a single ADQL expression.

        Parameters
        ----------
        pos_parts : list of str
            List of positive predicate strings.
        neg_parts : list of str
            List of negative predicate strings.

        Returns
        -------
        str
            The combined ADQL predicate.
        """
        pos_expr = ""
        if len(pos_parts) == 1:
            pos_expr = pos_parts[0]
        elif len(pos_parts) > 1:
            pos_expr = "(" + " OR ".join(pos_parts) + ")"

        if neg_parts and pos_expr:
            return "(" + " AND ".join(neg_parts) + ") AND " + pos_expr
        if neg_parts:
            return " AND ".join(neg_parts)
        return pos_expr

    def _build_list_predicate(self, col, pos_items, neg_items, pos_builder, neg_builder):
        """
        General helper to build predicates for list values with separate positive and negative handling.

        Parameters
        ----------
        col : str
            The column name.
        pos_items : list
            List of positive values.
        neg_items : list
            List of negative values.
        pos_builder : function
            Function to build positive predicates, called as pos_builder(col, pos_items).
        neg_builder : function
            Function to build negative predicates, called as neg_builder(col, neg_items).

        Returns
        -------
        str
            The combined ADQL predicate for the list values.
        """
        pos_parts = pos_builder(col, pos_items) if pos_items else []
        neg_parts = [neg_builder(col, v) for v in neg_items] if neg_items else []
        return self._combine_predicates(pos_parts, neg_parts)

    def _build_numeric_list_predicate(self, col, pos_items, neg_items):
        """
        Build predicate for multiple values passed into a numeric column with separated positives and negatives.

        Parameters
        ----------
        col : str
            The column name.
        pos_items : list
            List of positive values.
        neg_items : list
            List of negative values.

        Returns
        -------
        str
            The ADQL predicate for the numeric list.
        """
        def build_positive(col, items):
            # Separate simple numeric values from complex expressions (comparisons, ranges)
            simple_numbers = []
            complex_parts = []
            for val in items:
                if isinstance(val, bool):
                    simple_numbers.append(int(val))
                elif isinstance(val, (int, float)):
                    simple_numbers.append(val)
                elif isinstance(val, str):
                    try:
                        simple_numbers.append(float(val))
                    except ValueError:
                        complex_parts.append(self._parse_numeric_expr(col, val))
                else:
                    try:
                        simple_numbers.append(float(val))
                    except (ValueError, TypeError):
                        raise InvalidQueryError(f"Unsupported numeric value type: {type(val)}")

            parts = []
            if simple_numbers:
                vals = [str(v) for v in simple_numbers]
                parts.append(f"{col} IN (" + ", ".join(vals) + ")")
            parts.extend(complex_parts)
            return parts

        def build_negative(col, v):
            # For negatives, we can't use NOT IN or NOT (complex), so we build each as a separate predicate
            # and AND them together
            return self._format_scalar_predicate(col, f"!{v}", is_numeric=True)

        return self._build_list_predicate(
            col,
            pos_items,
            neg_items,
            build_positive,
            build_negative,
        )

    def _build_string_list_predicate(self, col, pos_items, neg_items):
        """
        Build predicate for multiple values passed into a string column with separated positives and negatives.

        Parameters
        ----------
        col : str
            The column name.
        pos_items : list
            List of positive values.
        neg_items : list
            List of negative values.

        Returns
        -------
        str
            The ADQL predicate for the string list.
        """
        def build_positive(col, items):
            # Separate simple strings (no wildcards) from those that require LIKE
            simple_strings = []
            pattern_parts = []
            for v in items:
                if isinstance(v, bool):
                    simple_strings.append(str(int(v)))
                elif isinstance(v, str):
                    if ("*" in v) or ("%" in v):
                        patt = self._quote_adql_string(v.replace("*", "%"))
                        pattern_parts.append(f"{col} LIKE '{patt}'")
                    else:
                        simple_strings.append("'" + self._quote_adql_string(v) + "'")
                else:
                    simple_strings.append(str(v))

            parts = []
            if simple_strings:
                parts.append(f"{col} IN (" + ", ".join(simple_strings) + ")")
            parts.extend(pattern_parts)
            return parts

        # Negative predicates → use helper
        def build_negative(col, v):
            # For negatives, we can't use NOT IN or NOT (LIKE), so we build each as a separate predicate
            # and AND them together
            return self._format_scalar_predicate(col, f"!{v}")

        return self._build_list_predicate(
            col,
            pos_items,
            neg_items,
            build_positive,
            build_negative,
        )

    def _build_temporal_list_predicate(self, col, pos_items, neg_items):
        """
        Build temporal list predicates using column datatype to choose CAST vs string comparison.

        Parameters
        ----------
        col : str
            The column name.
        pos_items : list
            List of positive values.
        neg_items : list
            List of negative values.

        Returns
        -------
        str
            The ADQL predicate for the temporal list.
        """
        return self._build_list_predicate(
            col,
            pos_items,
            neg_items,
            pos_builder=lambda c, vals: [self._parse_temporal_expr(c, v) for v in vals],
            neg_builder=lambda c, v: self._format_scalar_predicate(c, f"!{v}", is_temporal=True),
        )

    def _format_criteria_conditions(self, collection_obj, catalog, criteria, *, column_alias=None):
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
        column_meta = collection_obj.get_catalog_metadata(catalog).column_metadata
        numeric_cols, temporal_cols = self._classify_columns(column_meta)
        if column_alias is not None:
            numeric_cols = {name.lower() for name in numeric_cols}
            temporal_cols = {name.lower() for name in temporal_cols}
        conditions = []
        for key, value in criteria.items():
            column = f"{column_alias}.{key}" if column_alias is not None else key
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
                    if isinstance(v, str) and v.startswith("!"):
                        neg_items.append(v[1:].strip())
                    else:
                        pos_items.append(v)

                if key in temporal_cols:
                    expr = self._build_temporal_list_predicate(column, pos_items, neg_items)
                    if expr:
                        conditions.append(expr)
                elif key in numeric_cols:
                    expr = self._build_numeric_list_predicate(column, pos_items, neg_items)
                    if expr:
                        conditions.append(expr)
                else:
                    expr = self._build_string_list_predicate(column, pos_items, neg_items)
                    if expr:
                        conditions.append(expr)
            else:
                conditions.append(
                    self._format_scalar_predicate(column, value, key in numeric_cols, key in temporal_cols)
                )
        return conditions

    def _make_data_url(self, row):
        """Return the correct data URL for a given spectrum row."""
        dataset = row["DatasetName"]
        if row["SpectrumType"] < 2:
            return f"https://hla.stsci.edu/cgi-bin/getdata.cgi?config=ops&dataset={dataset}"
        return f"https://hla.stsci.edu/cgi-bin/ecfproxy?file_id={dataset}.fits"


Catalogs = CatalogsClass()
