# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
MAST Missions
=================

This module contains methods for searching MAST missions.
"""

import difflib
import numpy as np
import requests
import warnings
from pathlib import Path
from urllib.parse import quote

import astropy.units as u
import astropy.coordinates as coord
from astropy.table import Table, Row

from astroquery.utils import commons, async_to_sync
from astroquery.utils.class_or_instance import class_or_instance
from astroquery.exceptions import InputWarning, InvalidQueryError, MaxResultsWarning, NoResultsWarning

from . import conf, utils
from .core import MastQueryWithLogin

__all__ = ['MastMissionsClass', 'MastMissions']


@async_to_sync
class MastMissionsClass(MastQueryWithLogin):
    """
    MastMissions search class.
    Class that allows direct programmatic access to retrieve metadata via the MAST search API for a given mission.
    """

    # Static class variables
    _search = 'search'
    _list_products = 'list_products'
    _supported_missions = ['hst', 'jwst', 'classy', 'ullyses']

    def __init__(self, *, mission='hst'):
        super().__init__()

        self._search_option_fields = ['limit', 'offset', 'sort_by', 'search_key', 'sort_desc', 'select_cols',
                                      'skip_count', 'user_fields']
        self.dataset_kwds = {  # column keywords corresponding to dataset ID
            'hst': 'sci_data_set_name',
            'jwst': 'fileSetName'
        }
        self.limit = 5000  # maximum number of results
        self.columns = dict()
        self.service = self._search
        self.service_dict = {self._search: {'path': 'search'},
                             self._list_products: {'path': 'list_products'}}
        self.mission = mission
        self._service_api_connection.set_service_params(self.service_dict, f'search/{self.mission}')

    @property
    def mission(self):
        return self._mission

    @mission.setter
    def mission(self, value):
        # Check that mission is valid
        value = value.lower()
        if value not in self._supported_missions:
            raise ValueError(f'Mission {value} is not a supported mission.'
                             f' Mission must be one of: {", ".join(self._supported_missions)}')

        # Need to update the service parameters if the mission is changed
        self._mission = value
        self._service_api_connection.set_service_params(self.service_dict, f'search/{self.mission}')

    def _parse_result(self, response, *, verbose=False):  # Used by the async_to_sync decorator functionality
        """
        Parse the results of a `~requests.Response` objects and return an `~astropy.table.Table` of results.

        Parameters
        ----------
        response : `~requests.Response`
            `~requests.Response` objects.
        verbose : bool
            (presently does nothing - there is no output with verbose set to
            True or False)
            Default False. Setting to True provides more extensive output.

        Returns
        -------
        response : `~astropy.table.Table`
        """

        if self.service == self._search:
            results = self._service_api_connection._parse_result(response, verbose, data_key='results')
        elif self.service == self._list_products:
            # Results from list_products endpoint need to be handled differently
            results = Table(response.json()['products'])

        if len(results) >= self.limit:
            warnings.warn("Maximum results returned, may not include all sources within radius.",
                          MaxResultsWarning)

        return results

    def _validate_criteria(self, **criteria):
        """
        Check that criteria keyword arguments are valid key names. Raises InvalidQueryError if a criteria
        argument is not valid

        Parameters
        ----------
        **kwargs
            Keyword arguments representing criteria filters to apply.

        Raises
        -------
        InvalidQueryError
            If a keyword does not match any valid column names, an error is raised that suggests the closest
            matching column name, if available.
        """
        # Ensure that self.columns in populated
        self.get_column_list()

        # Check each criteria argument for validity
        valid_cols = self.columns[self.mission]['name']
        for kwd in criteria.keys():
            if kwd not in valid_cols:
                closest_match = difflib.get_close_matches(kwd, valid_cols, n=1)
                error_msg = (
                    f"Filter '{kwd}' does not exist. Did you mean '{closest_match[0]}'?"
                    if closest_match
                    else f"Filter '{kwd}' does not exist."
                )
                raise InvalidQueryError(error_msg)

    @class_or_instance
    def query_region_async(self, coordinates, *, radius=3*u.arcmin, limit=5000, offset=0,
                           select_cols=[], **kwargs):
        """
        Given a sky position and radius, returns a list of matching dataset IDs.

        Parameters
        ----------
        coordinates : str or `~astropy.coordinates` object
            The target around which to search. It may be specified as a
            string or as the appropriate `~astropy.coordinates` object.
        radius : str or `~astropy.units.Quantity` object, optional
            Default 3 degrees.
            The string must be parsable by `~astropy.coordinates.Angle`. The
            appropriate `~astropy.units.Quantity` object from
            `~astropy.units` may also be used. Defaults to 3 arcminutes.
        limit : int
            Optional and default is 5000.
            the maximum number of dataset IDs in the results.
        offset : int
            Optional and default is 0
            the number of records you wish to skip before selecting records.
        select_cols: list
            names of columns that will be included in the astropy table
        **kwargs
            Other mission-specific keyword args.
            All valid key names can be found using `~astroquery.mast.missions.MastMissionsClass.get_column_list`
            function.
            For example, one can specify the output columns(select_cols) or use other filters(conditions)

        Returns
        -------
        response : list of `~requests.Response`
        """

        self.limit = limit
        self.service = self._search

        # Check that criteria arguments are valid
        self._validate_criteria(**kwargs)

        # Put coordinates and radius into consistent format
        coordinates = commons.parse_coordinates(coordinates)

        # if radius is just a number we assume degrees
        radius = coord.Angle(radius, u.arcmin)

        # Dataset ID column should always be returned
        select_cols.append(self.dataset_kwds[self.mission])

        # basic params
        params = {'target': [f"{coordinates.ra.deg} {coordinates.dec.deg}"],
                  'radius': radius.arcsec,
                  'radius_units': 'arcseconds',
                  'limit': limit,
                  'offset': offset,
                  'select_cols': select_cols}

        params['conditions'] = []
        # adding additional user specified parameters
        for prop, value in kwargs.items():
            if prop not in self._search_option_fields:
                params['conditions'].append({prop: value})
            else:
                params[prop] = value

        return self._service_api_connection.missions_request_async(self.service, params)

    @class_or_instance
    def query_criteria_async(self, *, coordinates=None, objectname=None, radius=3*u.arcmin,
                             limit=5000, offset=0, select_cols=[], **criteria):
        """
        Given a set of search criteria, returns a list of mission metadata.

        Parameters
        ----------
        coordinates : str or `~astropy.coordinates` object
            The target around which to search. It may be specified as a
            string or as the appropriate `~astropy.coordinates` object.
        objectname : str
            The name of the target around which to search.
        radius : str or `~astropy.units.Quantity` object, optional
            Default 3 degrees.
            The string must be parsable by `~astropy.coordinates.Angle`. The
            appropriate `~astropy.units.Quantity` object from
            `~astropy.units` may also be used. Defaults to 3 arcminutes.
        limit : int
            Optional and default is 5000.
            the maximum number of dataset IDs in the results.
        offset : int
            Optional and default is 0.
            the number of records you wish to skip before selecting records.
        select_cols: list
            names of columns that will be included in the astropy table
        **criteria
            Criteria to apply. At least one non-positional criteria must be supplied.
            Valid criteria are coordinates, objectname, radius (as in
            `~astroquery.mast.missions.MastMissionsClass.query_region` and
            `~astroquery.mast.missions.MastMissionsClass.query_object` functions),
            and all fields listed in the column documentation for the mission being queried.
            Any invalid keys passed in criteria are ignored by the API.
            List of all valid fields that can be used to match results on criteria can be retrieved by calling
            `~astroquery.mast.missions.MastMissionsClass.get_column_list` function.

        Returns
        -------
        response : list of `~requests.Response`
        """

        self.limit = limit
        self.service = self._search

        # Check that criteria arguments are valid
        self._validate_criteria(**criteria)

        # Parse user input location
        if objectname or coordinates:
            coordinates = utils.parse_input_location(coordinates, objectname)

        # if radius is just a number we assume degrees
        radius = coord.Angle(radius, u.arcmin)

        # Dataset ID column should always be returned
        select_cols.append(self.dataset_kwds[self.mission])

        # build query
        params = {"limit": self.limit, "offset": offset, 'select_cols': select_cols}
        if coordinates:
            params["target"] = [f"{coordinates.ra.deg} {coordinates.dec.deg}"]
            params["radius"] = radius.arcsec
            params["radius_units"] = 'arcseconds'

        if not self._service_api_connection.check_catalogs_criteria_params(criteria):
            raise InvalidQueryError("At least one non-positional criterion must be supplied.")

        params['conditions'] = []
        for prop, value in criteria.items():
            if prop not in self._search_option_fields:
                params['conditions'].append({prop: value})
            else:
                params[prop] = value

        return self._service_api_connection.missions_request_async(self.service, params)

    @class_or_instance
    def query_object_async(self, objectname, *, radius=3*u.arcmin, limit=5000, offset=0, **kwargs):
        """
        Given an object name, returns a list of matching rows.

        Parameters
        ----------
        objectname : str
            The name of the target around which to search.
        radius : str or `~astropy.units.Quantity` object, optional
            Default 3 arcmin.
            The string must be parsable by `~astropy.coordinates.Angle`.
            The appropriate `~astropy.units.Quantity` object from
            `~astropy.units` may also be used. Defaults to 3 arcminutes.
        limit : int
            Optional and default is 5000.
            the maximum number of dataset IDs in the results.
        offset : int
            Optional and default is 0.
            the number of records you wish to skip before selecting records.
        **kwargs
            Mission-specific keyword args.
            Any invalid keys are ignored by the API.
            All valid keys can be found by calling `~astroquery.mast.missions.MastMissionsClass.get_column_list`
            function.

        Returns
        -------
        response : list of `~requests.Response`
        """

        coordinates = utils.resolve_object(objectname)

        return self.query_region_async(coordinates, radius=radius, limit=limit, offset=offset, **kwargs)

    @class_or_instance
    def get_product_list_async(self, datasets):
        """
        Given a dataset ID or list of dataset IDs, returns a list of associated data products.

        Parameters
        ----------
        datasets : str, list, `~astropy.table.Row`, `~astropy.table.Table`
            Row/Table of MastMissions query results (e.g. output from `query_object`)
            or single/list of dataset ID(s).

        Returns
        -------
        response : list of `~requests.Response`
        """

        self.service = self._list_products

        # Extract dataset IDs based on input type and mission
        if isinstance(datasets, Table):
            datasets = datasets[self.dataset_kwds[self.mission]]
        elif isinstance(datasets, Row):
            datasets = np.array([datasets[self.dataset_kwds[self.mission]]])
        elif isinstance(datasets, str):
            datasets = np.array([datasets])
        elif isinstance(datasets, list):
            datasets = np.array(datasets)
        else:
            raise TypeError('Unsupported data type for `datasets`. Expected string, '
                            'list of strings, Astropy row, or Astropy Table.')

        # Filter out empty strings from IDs
        datasets = datasets[datasets != '']
        if datasets.size == 0:
            raise InvalidQueryError("Dataset list is empty, no associated products.")

        # Send async service request
        params = {'dataset_ids': ','.join(datasets)}
        return self._service_api_connection.missions_request_async(self.service, params)

    def filter_products(self, products, *, extension=None, **filters):
        """
        Filters an `~astropy.table.Table` of mission data products based on given filters.

        Parameters
        ----------
        products : `~astropy.table.Table`
            Table containing data products to be filtered.
        extension : string or array, optional
            Default is None. Filters by file extension(s), matching any specified extensions.
        **filters :
            Column-based filters to be applied.
            Each keyword corresponds to a column name in the table, with the argument being one or more
            acceptable values for that column. AND logic is applied between filters, OR logic within
            each filter set.
            For example: type="science", extension=["fits","jpg"]

        Returns
        -------
        response : `~astropy.table.Table`
            Filtered Table of data products.
        """

        # Start with a mask of True for all entries
        filter_mask = np.full(len(products), True, dtype=bool)

        # Filter by file extension, if provided
        if extension:
            extensions = [extension] if isinstance(extension, str) else extension
            ext_mask = np.array(
                [not isinstance(x, np.ma.core.MaskedConstant) and any(x.endswith(ext) for ext in extensions)
                 for x in products["filename"]],
                dtype=bool
            )
            filter_mask &= ext_mask

        # Applying column-based filters
        for colname, vals in filters.items():
            if colname not in products.colnames:
                warnings.warn(f"Column '{colname}' not found in product table.", InputWarning)

            vals = [vals] if isinstance(vals, str) else vals
            col_mask = np.isin(products[colname], vals)
            filter_mask &= col_mask

        # Return filtered products
        return products[filter_mask]

    def download_file(self, uri, *, local_path=None, cache=True, verbose=True):
        """
        Downloads a single file based on the data URI.

        Parameters
        ----------
        uri : str
            The product dataURI
        local_path : str
            Directory or filename to which the file will be downloaded.  Defaults to current working directory.
        cache : bool
            Default is True. If file is found on disk, it will not be downloaded again.
        verbose : bool, optional
            Default is True. Whether to show download progress in the console.

        Returns
        -------
        status: str
            Download status message.  Either COMPLETE, SKIPPED, or ERROR.
        msg : str
            An error status message, if any.
        url : str
            The full URL download path.
        """

        # Construct the full data URL
        base_url = self._service_api_connection.MISSIONS_DOWNLOAD_URL + self.mission + '/api/v0.1/retrieve_product'
        data_url = base_url + '?product_name=' + uri
        escaped_url = base_url + '?product_name=' + quote(uri, safe=':')

        # Determine local file path. Use current directory as default.
        filename = Path(uri).name
        local_path = Path(local_path or filename)
        if not local_path.suffix:  # Append filename if local path is directory
            local_path = local_path / filename
            local_path.parent.mkdir(parents=True, exist_ok=True)

        status = 'COMPLETE'
        msg = None
        url = None

        try:
            # Attempt file download
            self._download_file(escaped_url, local_path, cache=cache, continuation=False, verbose=verbose)

            # Check if file exists
            if not local_path.is_file() and status != 'SKIPPED':
                status = 'ERROR'
                msg = 'File was not downloaded'
                url = data_url

        except requests.HTTPError as err:
            status = 'ERROR'
            msg = 'HTTPError: {0}'.format(err)
            url = data_url

        return status, msg, url

    def _download_files(self, products, base_dir, *, flat=False, cache=True, verbose=True):
        """
        Downloads files listed in an `~astropy.table.Table` of data products to a specified directory.

        Parameters
        ----------
        products : `~astropy.table.Table`
            Table containing products to be downloaded.
        base_dir : str
            Directory in which files will be downloaded.
        flat : bool
            Default is False.  If True, all files are downloaded directly to `base_dir`, and no subdirectories
            will be created.
        cache : bool
            Default is True. If file is found on disk, it will not be downloaded again.
        verbose : bool, optional
            Default is True. Whether to show download progress in the console.

        Returns
        -------
        response : `~astropy.table.Table`
            Table containing download results for each data product file.
        """

        manifest_entries = []
        base_dir = Path(base_dir)

        for data_product in products:

            # Determine local path for each file
            local_path = base_dir / data_product['dataset'] if not flat else base_dir
            local_path.mkdir(parents=True, exist_ok=True)
            local_file_path = local_path / Path(data_product['filename']).name

            # Download files and record status
            status, msg, url = self.download_file(data_product['uri'],
                                                  local_path=local_file_path,
                                                  cache=cache,
                                                  verbose=verbose)
            manifest_entries.append([local_path, status, msg or '', url or ''])

        manifest = Table(rows=manifest_entries, names=('Local Path', 'Status', 'Message', 'URL'))
        return manifest

    def download_products(self, products, *, download_dir=None, flat=False,
                          cache=True, extension=None, verbose=True, **filters):
        """
        Download specified data products.

        Parameters
        ----------
        products : str, list, `~astropy.table.Table`
            Either a single or list of dataset IDs (e.g., as input for `get_product_list`),
            or a Table of products (e.g., as output from `get_product_list`)
        download_dir : str or Path, optional
            Directory for file downloads.  Defaults to current directory.
        flat : bool, optional
            Default is False. If False, puts files into the standard
            directory structure of "mastDownload/<mission>/<dataset ID>/".
            If True, places files directly in `download_dir` without subdirectories.
        cache : bool, optional
            Default is True. If file is found on disc, it will not be downloaded again.
        extension : string or list, optional
            Default is None. Filter by file extension.
        verbose : bool, optional
            Default is True. Whether to show download progress in the console.
        **filters :
            Column-based filters to be applied.
            Each keyword corresponds to a column name in the table, with the argument being one or more
            acceptable values for that column. AND logic is applied between filters, OR logic within
            each filter set.
            For example: type="science", extension=["fits","jpg"]

        Returns
        -------
        manifest : `~astropy.table.Table`
            A table manifest showing downloaded file locations and statuses.
        """
        # Ensure `products` is a Table, collecting products if necessary
        if isinstance(products, (str, list)):
            products = [products] if isinstance(products, str) else products
            products = Table(np.vstack([self.get_product_list(oid) for oid in products]), masked=True)
        elif isinstance(products, Row):
            products = Table(products, masked=True)

        # Apply filters
        products = self.filter_products(products, extension=extension, **filters)

        # Remove duplicates
        products = utils.remove_duplicate_products(products, 'uri')

        if not len(products):
            warnings.warn("No products to download.", NoResultsWarning)
            return

        # Set up base directory for downloads
        download_dir = Path(download_dir or '.')
        base_dir = download_dir if flat else download_dir / 'mastDownload' / self.mission

        # Download files
        manifest = self._download_files(products,
                                        base_dir=base_dir,
                                        flat=flat,
                                        cache=cache,
                                        verbose=verbose)

        return manifest

    @class_or_instance
    def get_column_list(self):
        """
        For a mission, return a list of all searchable columns and their descriptions

        Returns
        -------
        response : `~astropy.table.Table` that contains columns names, types  and their descriptions
        """

        if not self.columns.get(self.mission):
            url = f"{conf.server}/search/util/api/v0.1/column_list?mission={self.mission}"

            try:
                resp = requests.get(url)
                resp.raise_for_status()

                # Parse JSON and extract necessary info
                results = resp.json()
                rows = [
                    (result['column_name'], result['qual_type'], result['description'])
                    for result in results
                ]

                # Create Table with parsed data
                col_table = Table(rows=rows, names=('name', 'data_type', 'description'))
                self.columns[self.mission] = col_table
            except Exception:
                raise Exception(f"Error occurred while trying to get column list for mission {self.mission}")

        return self.columns[self.mission]


MastMissions = MastMissionsClass()
