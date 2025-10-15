# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
MAST Collections
================

This module contains various methods for querying MAST collections such as catalogs.
"""

import difflib
from json import JSONDecodeError
import warnings
import os
import time

import astropy.units as u
import astropy.coordinates as coord
from astropy.table import Table, Row
from pyvo.dal import TAPService
from requests import HTTPError, RequestException
import requests

from ..utils import commons, async_to_sync
from ..utils.class_or_instance import class_or_instance
from ..exceptions import InvalidQueryError, MaxResultsWarning, InputWarning

from . import utils, conf
from .core import MastQueryWithLogin


__all__ = ['Catalogs', 'CatalogsClass']


@async_to_sync
class CatalogsClass(MastQueryWithLogin):
    """
    MAST catalog query class.

    Class for querying MAST catalog data.
    """

    def __init__(self, catalog="hsc", table=None):

        super().__init__()

        services = {"panstarrs": {"path": "panstarrs/{data_release}/{table}.json",
                                  "args": {"data_release": "dr2", "table": "mean"}}}
        self._catalogs_mast_search_options = ['columns', 'sort_by', 'table', 'data_release']

        self._service_api_connection.set_service_params(services, "catalogs", True)

        self.catalog_limit = None
        self._current_connection = None
        self._service_columns = dict()  # Info about columns for Catalogs.MAST services

        self._tap_base_url = "https://masttest.stsci.edu/vo-tap/api/v0.1/"
        self._tables_by_catalog_cache = dict()
        # Cache of pyvo TAPService instances per catalog (lowercased key)
        self._tap_services = {}
        self._catalogs_cache = self.get_catalogs()['catalog_name'].tolist()
        # Cache for column metadata fetched via TAP (keyed by (catalog, table))
        self._column_metadata_cache = {}

        self._catalog = None
        self._table = None
        self.catalog = catalog
        if table:
            self.table = table

    @property
    def catalog(self):
        return self._catalog

    @catalog.setter
    def catalog(self, catalog):
        catalog = catalog.lower()
        self._verify_catalog(catalog)

        # Update internal catalog and table
        self._catalog = catalog

        # Cache the table list for this catalog if not already done
        if catalog not in self._tables_by_catalog_cache:
            self._tables_by_catalog_cache[catalog] = self.get_tables(catalog)
        table_names = self._tables_by_catalog_cache[catalog]['table_name'].tolist()

        # Pick default table = first one that does NOT start with "tap_schema."
        default_table = next((t for t in table_names if not t.startswith("tap_schema.")), None)

        # If no valid table found, fallback to the first one
        if default_table is None:
            default_table = table_names[0] if table_names else None

        # Only change table if not set yet or invalid for this catalog
        if not hasattr(self, "_table") or self._table not in table_names:
            self._table = default_table

    @property    
    def table(self):
        return self._table
    
    @table.setter
    def table(self, table):
        # Setter that updates the service parameters if the table is changed
        self._verify_table(self.catalog, table)
        self._table = table


    def _parse_result(self, response, *, verbose=False):

        results_table = self._current_connection._parse_result(response, verbose=verbose)

        if len(results_table) == self.catalog_limit:
            warnings.warn("Maximum catalog results returned, may not include all sources within radius.",
                          MaxResultsWarning)

        return results_table
    
    def _verify_catalog(self, catalog):
        """
        Verify that the specified catalog is valid.

        Parameters
        ----------
        catalog : str
            The catalog to be verified.

        Raises
        ------
        ValueError
            If the specified catalog is not valid.
        """
        if catalog.lower() not in self._catalogs_cache:
            closest_match = difflib.get_close_matches(catalog, self._catalogs_cache, n=1)
            error_msg = (
                f"Catalog '{catalog}' is not recognized. Did you mean '{closest_match[0]}'?"
                if closest_match
                else f"Catalog '{catalog}' is not recognized."
            )
            error_msg += " Available catalogs are: " + ", ".join(self._catalogs_cache)
            raise ValueError(error_msg)
        
    def _verify_table(self, catalog, table):
        """
        Verify that the specified table is valid for the given catalog.

        Parameters
        ----------
        catalog : str
            The catalog to be verified.
        table : str
            The table to be verified.

        Raises
        ------
        ValueError
            If the specified table is not valid for the given catalog.
        """
        catalog = catalog.lower()
        if catalog not in self._tables_by_catalog_cache:
            self._tables_by_catalog_cache[catalog] = self.get_tables(catalog)

        table_names = self._tables_by_catalog_cache[catalog]['table_name'].tolist()
        lower_map = {name.lower(): name for name in table_names}
        if table.lower() not in lower_map:
            closest_match = difflib.get_close_matches(table, table_names, n=1)
            error_msg = (
                f"Table '{table}' is not recognized for catalog '{catalog}'. Did you mean '{closest_match[0]}'?"
                if closest_match
                else f"Table '{table}' is not recognized for catalog '{catalog}'."
            )
            error_msg += " Available tables are: " + ", ".join(table_names)
            raise ValueError(error_msg)
        
    def _verify_criteria(self, catalog, table, **criteria):
        """
        Check that criteria keyword arguments are valid column names for the specified catalog and table.

        Parameters
        ----------
        catalog : str
            The catalog to be queried.
        table : str
            The table within the catalog to query.
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
        column_metadata = self.get_column_metadata(catalog, table)
        col_names = list(column_metadata['name'])

        # Check each criteria argument for validity
        for kwd in criteria.keys():
            if kwd not in col_names:
                closest_match = difflib.get_close_matches(kwd, col_names, n=1)
                error_msg = (
                    f"Filter '{kwd}' is not recognized for catalog '{catalog}' and table '{table}'. Did you mean '{closest_match[0]}'?"
                    if closest_match
                    else f"Filter '{kwd}' is not recognized for catalog '{catalog}' and table '{table}'."
                )
                raise InvalidQueryError(error_msg)

    # ------------------------ Internal helpers for TAP querying ------------------------
    def _parse_inputs(self, catalog=None, table=None):
        """
        Return (catalog, table) applying default attributes, validation, and normalization.

        Parameters
        ----------
        catalog : str, optional
            The catalog to be queried. If None, uses the instance's default catalog.
        table : str, optional
            The table within the catalog to query. If None, uses the instance's default table.

        Returns
        -------
        tuple
            A tuple containing the (catalog, table) to be queried.
        """
        if not catalog:
            catalog = self.catalog
        else:
            catalog = catalog.lower()
            self._verify_catalog(catalog)

        if not table:
            table = self.table
        else:
            table = table.lower()
        #TODO: If the table is not present for the catalog, should we raise an error or default to a valid table? More relevant for default class attribute table
        self._verify_table(catalog, table)

        return catalog, table

    # ---- Formatting helpers extracted for readability ----
    def _get_numeric_columns(self, catalog, table):
        """Return a set of column names with numeric types for a given table.
        Relies on metadata types to detect numeric columns.
        """
        meta = self.get_column_metadata(catalog, table)
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
        """Parse numeric comparison/range string.
        Returns one of:
          - ('between', low, high)
          - ('cmp', op, num)
          - None if not a recognized pattern
        """
        if not isinstance(s, str):
            return None
        import re
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
            # non-numeric
            has_wild = ('*' in sval) or ('%' in sval)
            pattern = self._quote_sql_string(sval.replace('*', '%'))
            expr = f"{col} LIKE '{pattern}'" if has_wild else f"{col} = '{pattern}'"
            return f"NOT ({expr})" if is_neg else expr
        # numerics or others
        return f"{col} = {val}"

    def _build_numeric_list_predicate(self, col, pos_items, neg_items):
        """Build predicate for a numeric column list with separated positives and negatives."""
        # positives: split into simple numbers and complex expressions
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

        # negatives: NOT(complex) or != numeric
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

    def _run_tap_query(self, catalog, adql):
        """
        Run a TAP query against the specified catalog.

        Parameters
        ----------
        catalog : str
            The catalog to be queried.
        adql : str
            The ADQL query string.

        Returns
        -------
        response : `~astropy.table.Table`
            The result of the TAP query as an Astropy Table.
        """
        tap = self._get_tap_service(catalog)
        result = tap.search(adql)
        return result.to_table()

    def _get_tap_service(self, catalog):
        """Return a cached TAPService for a catalog (creates once per catalog)."""
        key = catalog.lower()
        svc = self._tap_services.get(key)
        if svc is None:
            print('Creating new TAPService for catalog:', key)
            svc = TAPService(self._tap_base_url + key)
            self._tap_services[key] = svc
        return svc

    def clear_tap_service_cache(self, catalog=None):
        """Clear cached TAPService instances. If catalog is provided, clear only that one."""
        if catalog is None:
            self._tap_services.clear()
        else:
            self._tap_services.pop(catalog.lower(), None)
    
    def get_catalogs(self):
        """
        Return a list of available Catalogs.MAST catalogs.

        Returns
        -------
        response : list of str
            A list of available Catalogs.MAST catalogs.
        """
        # If already cached, use it directly
        if getattr(self, "_catalogs_cache", None):
            return Table([self._catalogs_cache], names=('catalog_name',))
        
        # Otherwise, fetch from the TAP service
        url = "https://masttest.stsci.edu/vo-tap/api/v0.1/openapi.json"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        # Extract catalog enumeration
        catalog_enum = data["components"]["schemas"]["CatalogName"]["enum"]

        # Cache the results
        self._catalogs_cache = catalog_enum

        # Build an Astropy Table to hold the results
        catalog_table = Table([catalog_enum], names=('catalog_name',))
        return catalog_table

    def get_tables(self, catalog=None):
        """
        For a given Catalogs.MAST catalog, return a list of available tables.

        Parameters
        ----------
        catalog : str
            The catalog to be queried.

        Returns
        -------
        response : list of str
            A list of available tables for the specified catalog.
        """
        # If no catalog specified, use the class attribute
        if not catalog:
            catalog = self.catalog
        else:
            catalog = catalog.lower()
            self._verify_catalog(catalog)

        if catalog in self._tables_by_catalog_cache:
            return self._tables_by_catalog_cache[catalog]

        tap = self._get_tap_service(catalog)
        tables = tap.tables
        names = [t.name for t in tables]
        descriptions = [t.description for t in tables]
        
        # Create an Astropy Table to hold the results
        result_table = Table([names, descriptions], names=('table_name', 'description'))
        self._tables_by_catalog_cache[catalog] = result_table
        return result_table
    
    def get_column_metadata(self, catalog, table):
        """
        For a given Catalogs.MAST catalog and table, return metadata about the table.

        Parameters
        ----------
        catalog : str
            The catalog to be queried.
        table : str
            The table within the catalog to get metadata for.
        Returns
        -------
        response : `~astropy.table.Table`
            A table containing metadata about the specified table, including column names, data types, and descriptions.
        """
        catalog, table = self._parse_inputs(catalog, table)
        key = (catalog, table)
        if key in self._column_metadata_cache:
            return self._column_metadata_cache[key]

        tap = self._get_tap_service(catalog)
        tap_table = next((t for name, t in tap.tables.items() if name.lower() == table.lower()), None)

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
        metadata_table = Table([col_names, col_datatypes, col_units, col_ucds, col_descriptions],
                               names=('name', 'data_type', 'unit', 'ucd', 'description'))

        # Cache and return
        self._column_metadata_cache[key] = metadata_table
        return metadata_table
    
    def query_criteria_tap(self, catalog, table, coordinates=None, objectname=None, radius=0.2*u.deg, resolver=None, **criteria):
        """
        Query a Catalogs.MAST catalog table using criteria filters via TAP.

        Parameters
        ----------
        catalog : str
            The catalog to be queried.
        table : str
            The table within the catalog to query.
        coordinates : str or `~astropy.coordinates` object, optional
            The target around which to search. It may be specified as a string (e.g., '350 -80') or as an Astropy coordinates object.
        objectname : str, optional
            The name of the object to resolve and search around.
        radius : str or `~astropy.units.Quantity` object, optional
            The search radius around the target coordinates or object. Default 0.2 degrees.
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
        catalog, table = self._parse_inputs(catalog, table)
        self._verify_criteria(catalog, table, **criteria)

        # If positional info supplied delegate to region query for DRYness
        if objectname or coordinates:
            coordinates = utils.parse_input_location(coordinates=coordinates,
                                                     objectname=objectname,
                                                     resolver=resolver)
            return self.query_region_tap(coordinates, radius=radius, catalog=catalog, table=table, **criteria)

        adql = f'SELECT * FROM {table} '
        if criteria:
            conditions = self._format_criteria_conditions(catalog, table, criteria)
            adql += 'WHERE ' + ' AND '.join(conditions)
        return self._run_tap_query(catalog, adql)
    
    def query_region_tap(self, coordinates, radius=0.2*u.deg, catalog='tic', table="dbo.catalogrecord", **criteria):
        catalog, table = self._parse_inputs(catalog, table)
        self._verify_criteria(catalog, table, **criteria)

        # Add positional constraint
        coordinates = commons.parse_coordinates(coordinates, return_frame='icrs')
        radius = coord.Angle(radius, u.deg)  # If radius is just a number we assume degrees
        adql = (f'SELECT * FROM {table.lower()} WHERE CONTAINS(POINT(\'ICRS\', ra, dec), '
                f'CIRCLE(\'ICRS\', {coordinates.ra.deg}, {coordinates.dec.deg}, {radius.to(u.deg).value})) = 1 ')

        # Add additional constraints
        if criteria:
            conditions = self._format_criteria_conditions(catalog, table, criteria)
            adql += 'AND ' + ' AND '.join(conditions)

        return self._run_tap_query(catalog, adql)
    
    def query_object_tap(self, objectname, radius=0.2*u.deg, catalog='tic', table="dbo.catalogrecord", resolver=None, **criteria):
        self._verify_catalog(catalog)
        self._verify_table(catalog, table)
        self._verify_criteria(catalog, table, **criteria)
        coordinates = utils.resolve_object(objectname, resolver=resolver)
        return self.query_region_tap(coordinates, radius=radius, catalog=catalog, table=table, **criteria)

    def _get_service_col_config(self, catalog, release='dr2', table='mean'):
        """
        For a given Catalogs.MAST catalog, return a list of all searchable columns and their descriptions.
        As of now, this function is exclusive to the Pan-STARRS catalog.

        Parameters
        ----------
        catalog : str
            The catalog to be queried.
        release : str, optional
            Catalog data release to query from.
        table : str, optional
            Catalog table to query from.

        Returns
        -------
        response : `~astropy.table.Table` that contains columns names, types, and descriptions
        """
        # Only supported for PanSTARRS currently
        if catalog != 'panstarrs':
            return

        service_key = (catalog, release, table)
        if service_key not in self._service_columns:
            try:
                # Send server request to get column list for given parameters
                request_url = f'{conf.catalogs_server}/api/v0.1/{catalog}/{release}/{table}/metadata.json'
                resp = utils._simple_request(request_url)

                # Parse JSON and extract necessary info
                results = resp.json()
                rows = [
                    (result['column_name'], result['db_type'], result['description'])
                    for result in results
                ]

                # Create Table with parsed data
                col_table = Table(rows=rows, names=('name', 'data_type', 'description'))
                self._service_columns[service_key] = col_table

            except JSONDecodeError as ex:
                raise JSONDecodeError(f'Failed to decode JSON response while attempting to get column list'
                                      f' for {catalog} catalog {table}, {release}: {ex}')
            except RequestException as ex:
                raise ConnectionError(f'Failed to connect to the server while attempting to get column list'
                                      f' for {catalog} catalog {table}, {release}: {ex}')
            except KeyError as ex:
                raise KeyError(f'Expected key not found in response data while attempting to get column list'
                               f' for {catalog} catalog {table}, {release}: {ex}')
            except Exception as ex:
                raise RuntimeError(f'An unexpected error occurred while attempting to get column list'
                                   f' for {catalog} catalog {table}, {release}: {ex}')

        return self._service_columns[service_key]

    def _validate_service_criteria(self, catalog, **criteria):
        """
        Check that criteria keyword arguments are valid column names for the service.
        Raises InvalidQueryError if a criteria argument is invalid.

        Parameters
        ----------
        catalog : str
            The catalog to be queried.
        **criteria
            Keyword arguments representing criteria filters to apply.

        Raises
        -------
        InvalidQueryError
            If a keyword does not match any valid column names, an error is raised that suggests the closest
            matching column name, if available.
        """
        # Ensure that self._service_columns is populated
        release = criteria.get('data_release', 'dr2')
        table = criteria.get('table', 'mean')
        col_config = self._get_service_col_config(catalog, release, table)

        if col_config:
            # Check each criteria argument for validity
            valid_cols = list(col_config['name']) + self._catalogs_mast_search_options
            for kwd in criteria.keys():
                col = next((name for name in valid_cols if name.lower() == kwd.lower()), None)
                if not col:
                    closest_match = difflib.get_close_matches(kwd, valid_cols, n=1)
                    error_msg = (
                        f"Filter '{kwd}' does not exist for {catalog} catalog {table}, {release}. "
                        f"Did you mean '{closest_match[0]}'?"
                        if closest_match
                        else f"Filter '{kwd}' does not exist for {catalog} catalog {table}, {release}."
                    )
                    raise InvalidQueryError(error_msg)

    @class_or_instance
    def query_region_async(self, coordinates, *, radius=0.2*u.deg, catalog="Hsc",
                           version=None, pagesize=None, page=None, **criteria):
        """
        Given a sky position and radius, returns a list of catalog entries.
        See column documentation for specific catalogs `here <https://mast.stsci.edu/api/v0/pages.html>`__.

        Parameters
        ----------
        coordinates : str or `~astropy.coordinates` object
            The target around which to search. It may be specified as a
            string or as the appropriate `~astropy.coordinates` object.
        radius : str or `~astropy.units.Quantity` object, optional
            Default 0.2 degrees.
            The string must be parsable by `~astropy.coordinates.Angle`. The
            appropriate `~astropy.units.Quantity` object from
            `~astropy.units` may also be used. Defaults to 0.2 deg.
        catalog : str, optional
            Default HSC.
            The catalog to be queried.
        version : int, optional
            Version number for catalogs that have versions. Default is highest version.
        pagesize : int, optional
            Default None.
            Can be used to override the default pagesize for (set in configs) this query only.
            E.g. when using a slow internet connection.
        page : int, optional
            Default None.
            Can be used to override the default behavior of all results being returned to obtain a
            specific page of results.
        **criteria
            Other catalog-specific keyword args.
            These can be found in the (service documentation)[https://mast.stsci.edu/api/v0/_services.html]
            for specific catalogs. For example, one can specify the magtype for an HSC search.
            For catalogs available through Catalogs.MAST (PanSTARRS), the Column Name is the keyword, and the argument
            should be either an acceptable value for that parameter, or a list consisting values, or  tuples of
            decorator, value pairs (decorator, value). In addition, columns may be used to select the return columns,
            consisting of a list of column names. Results may also be sorted through the query with the parameter
            sort_by composed of either a single Column Name to sort ASC, or a list of Column Nmaes to sort ASC or
            tuples of Column Name and Direction (ASC, DESC) to indicate sort order (Column Name, DESC).
            Detailed information of Catalogs.MAST criteria usage can
            be found `here <https://catalogs.mast.stsci.edu/docs/index.html>`__.

        Returns
        -------
        response : list of `~requests.Response`
        """

        # Put coordinates and radius into consistent format
        coordinates = commons.parse_coordinates(coordinates, return_frame='icrs')

        # if radius is just a number we assume degrees
        radius = coord.Angle(radius, u.deg)

        # basic params
        params = {'ra': coordinates.ra.deg,
                  'dec': coordinates.dec.deg,
                  'radius': radius.deg}

        # Determine API connection and service name
        if catalog.lower() in self._service_api_connection.SERVICES:
            self._current_connection = self._service_api_connection
            service = catalog

            # validate user criteria
            self._validate_service_criteria(catalog.lower(), **criteria)

            # adding additional user specified parameters
            for prop, value in criteria.items():
                params[prop] = value

        else:
            self._current_connection = self._portal_api_connection

            # valid criteria keywords
            valid_criteria = []

            # Sorting out the non-standard portal service names
            if catalog.lower() == "hsc":
                if version == 2:
                    service = "Mast.Hsc.Db.v2"
                else:
                    if version not in (3, None):
                        warnings.warn("Invalid HSC version number, defaulting to v3.", InputWarning)
                    service = "Mast.Hsc.Db.v3"

                # Hsc specific parameters (can be overridden by user)
                self.catalog_limit = criteria.pop('nr', 50000)
                valid_criteria = ['nr', 'ni', 'magtype']
                params['nr'] = self.catalog_limit
                params['ni'] = criteria.pop('ni', 1)
                params['magtype'] = criteria.pop('magtype', 1)

            elif catalog.lower() == "galex":
                service = "Mast.Galex.Catalog"
                self.catalog_limit = criteria.get('maxrecords', 50000)

                # galex specific parameters (can be overridden by user)
                valid_criteria = ['maxrecords']
                params['maxrecords'] = criteria.pop('maxrecords', 50000)

            elif catalog.lower() == "gaia":
                if version == 1:
                    service = "Mast.Catalogs.GaiaDR1.Cone"
                else:
                    if version not in (None, 2):
                        warnings.warn("Invalid Gaia version number, defaulting to DR2.", InputWarning)
                    service = "Mast.Catalogs.GaiaDR2.Cone"

            elif catalog.lower() == 'plato':
                if version in (None, 1):
                    service = "Mast.Catalogs.Plato.Cone"
                else:
                    warnings.warn("Invalid PLATO catalog version number, defaulting to DR1.", InputWarning)
                    service = "Mast.Catalogs.Plato.Cone"

            else:
                service = "Mast.Catalogs." + catalog + ".Cone"
                self.catalog_limit = None

            # additional user-specified parameters are not valid
            if criteria:
                key = next(iter(criteria))
                closest_match = difflib.get_close_matches(key, valid_criteria, n=1)
                error_msg = (
                    f"Filter '{key}' does not exist for catalog {catalog}. Did you mean '{closest_match[0]}'?"
                    if closest_match
                    else f"Filter '{key}' does not exist for catalog {catalog}."
                )
                raise InvalidQueryError(error_msg)

        # Parameters will be passed as JSON objects only when accessing the PANSTARRS API
        use_json = catalog.lower() == 'panstarrs'

        return self._current_connection.service_request_async(service, params, pagesize=pagesize, page=page,
                                                              use_json=use_json)

    @class_or_instance
    def query_object_async(self, objectname, *, radius=0.2*u.deg, catalog="Hsc",
                           pagesize=None, page=None, version=None, resolver=None, **criteria):
        """
        Given an object name, returns a list of catalog entries.
        See column documentation for specific catalogs `here <https://mast.stsci.edu/api/v0/pages.html>`__.

        Parameters
        ----------
        objectname : str
            The name of the target around which to search.
        radius : str or `~astropy.units.Quantity` object, optional
            Default 0.2 degrees.
            The string must be parsable by `~astropy.coordinates.Angle`.
            The appropriate `~astropy.units.Quantity` object from
            `~astropy.units` may also be used. Defaults to 0.2 deg.
        catalog : str, optional
            Default HSC.
            The catalog to be queried.
        pagesize : int, optional
            Default None.
            Can be used to override the default pagesize for (set in configs) this query only.
            E.g. when using a slow internet connection.
        page : int, optional
            Default None.
            Can be used to override the default behavior of all results being returned
            to obtain a specific page of results.
        version : int, optional
            Version number for catalogs that have versions. Default is highest version.
        resolver : str, optional
            The resolver to use when resolving a named target into coordinates. Valid options are "SIMBAD" and "NED".
            If not specified, the default resolver order will be used. Please see the
            `STScI Archive Name Translation Application (SANTA) <https://mastresolver.stsci.edu/Santa-war/>`__
            for more information. Default is None.
        **criteria
            Catalog-specific keyword args.
            These can be found in the `service documentation <https://mast.stsci.edu/api/v0/_services.html>`__.
            for specific catalogs. For example, one can specify the magtype for an HSC search.
            For catalogs available through Catalogs.MAST (PanSTARRS), the Column Name is the keyword, and the argument
            should be either an acceptable value for that parameter, or a list consisting values, or  tuples of
            decorator, value pairs (decorator, value). In addition, columns may be used to select the return columns,
            consisting of a list of column names. Results may also be sorted through the query with the parameter
            sort_by composed of either a single Column Name to sort ASC, or a list of Column Nmaes to sort ASC or
            tuples of Column Name and Direction (ASC, DESC) to indicate sort order (Column Name, DESC).
            Detailed information of Catalogs.MAST criteria usage can
            be found `here <https://catalogs.mast.stsci.edu/docs/index.html>`__.

        Returns
        -------
        response : list of `~requests.Response`
        """

        coordinates = utils.resolve_object(objectname, resolver=resolver)

        return self.query_region_async(coordinates,
                                       radius=radius,
                                       catalog=catalog,
                                       version=version,
                                       pagesize=pagesize,
                                       page=page,
                                       **criteria)

    @class_or_instance
    def query_criteria_async(self, catalog, *, pagesize=None, page=None, resolver=None, **criteria):
        """
        Given an set of filters, returns a list of catalog entries.
        See column documentation for specific catalogs `here <https://mast.stsci.edu/api/v0/pages.html>`__.

        Parameters
        ----------
        catalog : str
            The catalog to be queried.
        pagesize : int, optional
            Can be used to override the default pagesize.
            E.g. when using a slow internet connection.
        page : int, optional
            Can be used to override the default behavior of all results being returned to obtain
            one specific page of results.
        resolver : str, optional
            The resolver to use when resolving a named target into coordinates. Valid options are "SIMBAD" and "NED".
            If not specified, the default resolver order will be used. Please see the
            `STScI Archive Name Translation Application (SANTA) <https://mastresolver.stsci.edu/Santa-war/>`__
            for more information. Default is None.
        **criteria
            Criteria to apply. At least one non-positional criteria must be supplied.
            Valid criteria are coordinates, objectname, radius (as in `query_region` and `query_object`),
            and all fields listed in the column documentation for the catalog being queried.
            The Column Name is the keyword, with the argument being one or more acceptable values for that parameter,
            except for fields with a float datatype where the argument should be in the form [minVal, maxVal].
            For non-float type criteria wildcards maybe used (both * and % are considered wildcards), however
            only one wildcarded value can be processed per criterion.
            RA and Dec must be given in decimal degrees, and datetimes in MJD.
            For example: filters=["FUV","NUV"],proposal_pi="Ost*",t_max=[52264.4586,54452.8914]
            For catalogs available through Catalogs.MAST (PanSTARRS), the Column Name is the keyword, and the argument
            should be either an acceptable value for that parameter, or a list consisting values, or  tuples of
            decorator, value pairs (decorator, value). In addition, columns may be used to select the return columns,
            consisting of a list of column names. Results may also be sorted through the query with the parameter
            sort_by composed of either a single Column Name to sort ASC, or a list of Column Nmaes to sort ASC or
            tuples of Column Name and Direction (ASC, DESC) to indicate sort order (Column Name, DESC).
            Detailed information of Catalogs.MAST criteria usage can
            be found `here <https://catalogs.mast.stsci.edu/docs/index.html>`__.

        Returns
        -------
        response : list of `~requests.Response`
        """

        # Separating any position info from the rest of the filters
        coordinates = criteria.pop('coordinates', None)
        objectname = criteria.pop('objectname', None)
        radius = criteria.pop('radius', 0.2*u.deg)

        if objectname or coordinates:
            coordinates = utils.parse_input_location(coordinates=coordinates,
                                                     objectname=objectname,
                                                     resolver=resolver)

        # if radius is just a number we assume degrees
        radius = coord.Angle(radius, u.deg)

        # build query
        params = {}
        if coordinates:
            params["ra"] = coordinates.ra.deg
            params["dec"] = coordinates.dec.deg
            params["radius"] = radius.deg

        # Determine API connection, service name, and build filter set
        filters = None
        if catalog.lower() in self._service_api_connection.SERVICES:
            self._current_connection = self._service_api_connection
            service = catalog

            # validate user criteria
            self._validate_service_criteria(catalog.lower(), **criteria)

            if not self._current_connection.check_catalogs_criteria_params(criteria):
                raise InvalidQueryError("At least one non-positional criterion must be supplied.")

            for prop, value in criteria.items():
                params[prop] = value

        else:
            self._current_connection = self._portal_api_connection

            if catalog.lower() == "tic":
                service = "Mast.Catalogs.Filtered.Tic"
                if coordinates or objectname:
                    service += ".Position"
                service += ".Rows"  # Using the rowstore version of the query for speed
                column_config_name = "Mast.Catalogs.Tess.Cone"
                params["columns"] = "*"
            elif catalog.lower() == "ctl":
                service = "Mast.Catalogs.Filtered.Ctl"
                if coordinates or objectname:
                    service += ".Position"
                service += ".Rows"  # Using the rowstore version of the query for speed
                column_config_name = "Mast.Catalogs.Tess.Cone"
                params["columns"] = "*"
            elif catalog.lower() == "diskdetective":
                service = "Mast.Catalogs.Filtered.DiskDetective"
                if coordinates or objectname:
                    service += ".Position"
                column_config_name = "Mast.Catalogs.Dd.Cone"
            else:
                raise InvalidQueryError("Criteria query not available for {}".format(catalog))

            filters = self._current_connection.build_filter_set(column_config_name, service, **criteria)

            if not filters:
                raise InvalidQueryError("At least one non-positional criterion must be supplied.")
            params["filters"] = filters

        # Parameters will be passed as JSON objects only when accessing the PANSTARRS API
        use_json = catalog.lower() == 'panstarrs'

        return self._current_connection.service_request_async(service, params, pagesize=pagesize, page=page,
                                                              use_json=use_json)

    @class_or_instance
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
            match = match["MatchID"]
        match = str(match)  # np.int64 gives json serializer problems, so stringify right here

        if version == 2:
            service = "Mast.HscMatches.Db.v2"
        else:
            if version not in (3, None):
                warnings.warn("Invalid HSC version number, defaulting to v3.", InputWarning)
            service = "Mast.HscMatches.Db.v3"

        params = {"input": match}

        return self._current_connection.service_request_async(service, params, pagesize=pagesize, page=page)

    @class_or_instance
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

        service = "Mast.HscSpectra.Db.All"
        params = {}

        return self._current_connection.service_request_async(service, params, pagesize, page)

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

        # if spectra is not a Table, put it in a list
        if isinstance(spectra, Row):
            spectra = [spectra]

        # set up the download directory and paths
        if not download_dir:
            download_dir = '.'

        if curl_flag:  # don't want to download the files now, just the curl script

            download_file = "mastDownload_" + time.strftime("%Y%m%d%H%M%S")

            url_list = []
            path_list = []
            for spec in spectra:
                if spec['SpectrumType'] < 2:
                    url_list.append('https://hla.stsci.edu/cgi-bin/getdata.cgi?config=ops&dataset={0}'
                                    .format(spec['DatasetName']))

                else:
                    url_list.append('https://hla.stsci.edu/cgi-bin/ecfproxy?file_id={0}'
                                    .format(spec['DatasetName']) + '.fits')

                path_list.append(download_file + "/HSC/" + spec['DatasetName'] + '.fits')

            description_list = [""]*len(spectra)
            producttype_list = ['spectrum']*len(spectra)

            service = "Mast.Bundle.Request"
            params = {"urlList": ",".join(url_list),
                      "filename": download_file,
                      "pathList": ",".join(path_list),
                      "descriptionList": list(description_list),
                      "productTypeList": list(producttype_list),
                      "extension": 'curl'}

            response = self._portal_api_connection.service_request_async(service, params)
            bundler_response = response[0].json()

            local_path = os.path.join(download_dir, "{}.sh".format(download_file))
            self._download_file(bundler_response['url'], local_path, head_safe=True, continuation=False)

            status = "COMPLETE"
            msg = None
            url = None

            if not os.path.isfile(local_path):
                status = "ERROR"
                msg = "Curl could not be downloaded"
                url = bundler_response['url']
            else:
                missing_files = [x for x in bundler_response['statusList'].keys()
                                 if bundler_response['statusList'][x] != 'COMPLETE']
                if len(missing_files):
                    msg = "{} files could not be added to the curl script".format(len(missing_files))
                    url = ",".join(missing_files)

            manifest = Table({'Local Path': [local_path],
                              'Status': [status],
                              'Message': [msg],
                              "URL": [url]})

        else:
            base_dir = download_dir.rstrip('/') + "/mastDownload/HSC"

            if not os.path.exists(base_dir):
                os.makedirs(base_dir)

            manifest_array = []
            for spec in spectra:

                if spec['SpectrumType'] < 2:
                    data_url = f'https://hla.stsci.edu/cgi-bin/getdata.cgi?config=ops&dataset={spec["DatasetName"]}'
                else:
                    data_url = f'https://hla.stsci.edu/cgi-bin/ecfproxy?file_id={spec["DatasetName"]}.fits'

                local_path = os.path.join(base_dir, f'{spec["DatasetName"]}.fits')

                status = "COMPLETE"
                msg = None
                url = None

                try:
                    self._download_file(data_url, local_path, cache=cache, head_safe=True)

                    # check file size also this is where would perform md5
                    if not os.path.isfile(local_path):
                        status = "ERROR"
                        msg = "File was not downloaded"
                        url = data_url

                except HTTPError as err:
                    status = "ERROR"
                    msg = "HTTPError: {0}".format(err)
                    url = data_url

                manifest_array.append([local_path, status, msg, url])

                manifest = Table(rows=manifest_array, names=('Local Path', 'Status', 'Message', "URL"))

        return manifest


Catalogs = CatalogsClass()
