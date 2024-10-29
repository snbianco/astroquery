# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
MAST Missions
=================

This module contains methods for searching MAST missions.
"""

from urllib.parse import quote
import numpy as np
import requests
import warnings
from pathlib import Path

from astropy.table import Table, Row
import astropy.units as u
import astropy.coordinates as coord

from astroquery.utils import commons, async_to_sync
from astroquery.utils.class_or_instance import class_or_instance
from astroquery.exceptions import InvalidQueryError, MaxResultsWarning, NoResultsWarning

from astroquery.mast import utils
from astroquery.mast.core import MastQueryWithLogin

from . import conf

__all__ = ['MastMissionsClass', 'MastMissions']


@async_to_sync
class MastMissionsClass(MastQueryWithLogin):
    """
    MastMissions search class.
    Class that allows direct programmatic access to retrieve metadata via the MAST search API for a given mission.
    """

    def __init__(self, *, mission='hst'):
        super().__init__()

        self._search_option_fields = ['limit', 'offset', 'sort_by', 'search_key', 'sort_desc', 'select_cols',
                                      'skip_count', 'user_fields']
        self.mission = mission
        self.mission_kwds = {
            'hst': {'dataset_id': 'sci_data_set_name'},
            'jwst': {'dataset_id': 'fileSetName'}
        }
        self.limit = 5000

        self.service = 'search'
        service_dict = {'search': {'path': 'search', 'args': {}},
                        'list_products': {'path': 'list_products', 'args': {}},
                        'retrieve_product': {'path': 'retrieve_product', 'args': {}}}
        self._service_api_connection.set_service_params(service_dict, f'search/{self.mission}')

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

        if self.service == 'search':
            results = self._service_api_connection._parse_result(response, verbose, data_key='results')
        elif self.service == 'list_products':
            results = Table(response.json()['products'])

        if len(results) >= self.limit:
            warnings.warn("Maximum results returned, may not include all sources within radius.",
                          MaxResultsWarning)

        return results

    @class_or_instance
    def query_region_async(self, coordinates, *, radius=3*u.arcmin, limit=5000, offset=0, **kwargs):
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
        **kwargs
            Other mission-specific keyword args.
            Any invalid keys are ignored by the API.
            All valid key names can be found using `~astroquery.mast.missions.MastMissionsClass.get_column_list`
            function.
            For example one can specify the output columns(select_cols) or use other filters(conditions)

        Returns
        -------
        response : list of `~requests.Response`
        """

        self.limit = limit
        self.service = 'search'

        # Put coordinates and radius into consistent format
        coordinates = commons.parse_coordinates(coordinates)

        # if radius is just a number we assume degrees
        radius = coord.Angle(radius, u.arcmin)

        # basic params
        params = {'target': [f"{coordinates.ra.deg} {coordinates.dec.deg}"],
                  'radius': radius.arcsec,
                  'radius_units': 'arcseconds',
                  'limit': limit,
                  'offset': offset}

        params['conditions'] = []
        # adding additional user specified parameters
        for prop, value in kwargs.items():
            if prop not in self._search_option_fields:
                params['conditions'].append({prop: value})
            else:
                params[prop] = value

        return self._service_api_connection.service_request_async(self.service, params, use_json=True)

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
        self.service = 'search'

        if objectname or coordinates:
            coordinates = utils.parse_input_location(coordinates, objectname)

        # if radius is just a number we assume degrees
        radius = coord.Angle(radius, u.arcmin)

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

        return self._service_api_connection.service_request_async(self.service, params, use_json=True)

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
    def get_prodict_list_async(self, datasets):
        """
        Test
        """
        self.service = 'list_products'

        # Getting array of dataset IDs
        if isinstance(datasets, Table) or isinstance(datasets, Row):
            datasets = datasets[self.mission_kwds[self.mission]['dataset_id']]
        if isinstance(datasets, str):
            datasets = np.array([datasets])
        if isinstance(datasets, list):
            datasets = np.array(datasets)

        datasets = datasets[datasets != '']
        if datasets.size == 0:
            raise InvalidQueryError("Dataset list is empty, no associated products.")

        params = {'dataset_ids': ','.join(datasets)}

        return self._service_api_connection.service_request_async(self.service, params, method='GET', use_json=True)

    def filter_products(self, products, *, extension=None, **filters):
        """
        Takes an `~astropy.table.Table` of mission data products and filters it based on given filters.

        Parameters
        ----------
        products : `~astropy.table.Table`
            Table containing data products to be filtered.
        extension : string or array, optional
            Default None. Option to filter by file extension.
        **filters :
            Filters to be applied.
            The column name is the keyword, with the argument being one or more acceptable values
            for that parameter.
            Filter behavior is AND between the filters and OR within a filter set.
            For example: type="science", extension=["fits","jpg"]

        Returns
        -------
        response : `~astropy.table.Table`
        """

        filter_mask = np.full(len(products), True, dtype=bool)

        # Applying extension filter
        if extension:
            if isinstance(extension, str):
                extension = [extension]

            mask = np.full(len(products), False, dtype=bool)
            for elt in extension:
                mask |= [False if isinstance(x, np.ma.core.MaskedConstant) else x.endswith(elt)
                         for x in products["filename"]]
            filter_mask &= mask

        # Applying the rest of the filters
        for colname, vals in filters.items():

            if isinstance(vals, str):
                vals = [vals]

            mask = np.full(len(products), False, dtype=bool)
            for elt in vals:
                mask |= (products[colname] == elt)

            filter_mask &= mask

        return products[np.where(filter_mask)]

    def download_file(self, uri, *, local_path=None, cache=True, verbose=True):
        """
        Downloads a single file based on the data URI

        Parameters
        ----------
        uri : str
            The product dataURI, e.g. mast:JWST/product/jw00736-o039_t001_miri_ch1-long_x1d.fits
        local_path : str
            Directory or filename to which the file will be downloaded.  Defaults to current working directory.
        cache : bool
            Default is True. If file is found on disk it will not be downloaded again.
        verbose : bool, optional
            Default True. Whether to show download progress in the console.

        Returns
        -------
        status: str
            download status message.  Either COMPLETE, SKIPPED, or ERROR.
        msg : str
            An error status message, if any.
        url : str
            The full url download path
        """

        # create the full data URL
        base_url = self._service_api_connection.MISSIONS_DOWNLOAD_URL + self.mission + '/api/v0.1/retrieve_product'
        data_url = base_url + "?product_name=" + uri
        escaped_url = base_url + "?product_name=" + quote(uri, safe=":")

        # parse a local file path from local_path parameter.  Use current directory as default.
        filename = Path(uri).name
        if not local_path:  # local file path is not defined
            local_path = filename
        else:
            path = Path(local_path)
            if not path.suffix:  # local_path is a directory
                local_path = path / filename  # append filename
                if not path.exists():  # create directory if it doesn't exist
                    path.mkdir(parents=True, exist_ok=True)

        status = "COMPLETE"
        msg = None
        url = None

        try:
            self._download_file(escaped_url, local_path,
                                cache=cache, continuation=False,
                                verbose=verbose)

            # check if file exists also this is where would perform md5,
            # and also check the filesize if the database reliably reported file sizes
            if (not Path(local_path).is_file()) and (status != "SKIPPED"):
                status = "ERROR"
                msg = "File was not downloaded"
                url = data_url

        except requests.HTTPError as err:
            status = "ERROR"
            msg = "HTTPError: {0}".format(err)
            url = data_url

        return status, msg, url

    def _download_files(self, products, base_dir, *, flat=False, cache=True, verbose=True):
        """
        Takes an `~astropy.table.Table` of data products and downloads them into the directory given by base_dir.

        Parameters
        ----------
        products : `~astropy.table.Table`
            Table containing products to be downloaded.
        base_dir : str
            Directory in which files will be downloaded.
        flat : bool
            Default is False.  If set to True, no subdirectories will be made for the
            downloaded files.
        cache : bool
            Default is True. If file is found on disk it will not be downloaded again.
        verbose : bool, optional
            Default True. Whether to show download progress in the console.

        Returns
        -------
        response : `~astropy.table.Table`
        """

        manifest_array = []
        for data_product in products:

            # create the local file download path
            if not flat:
                local_path = Path(base_dir, data_product['dataset'])
                local_path.mkdir(parents=True, exist_ok=True)
            else:
                local_path = base_dir
            local_path = Path(local_path) / Path(data_product['filename']).name

            # download the files
            status, msg, url = self.download_file(data_product['uri'], local_path=local_path,
                                                  cache=cache, verbose=verbose)

            manifest_array.append([local_path, status, msg, url])

        manifest = Table(rows=manifest_array, names=('Local Path', 'Status', 'Message', "URL"))

        return manifest

    def download_products(self, products, *, download_dir=None, flat=False,
                          cache=True, extension=None, verbose=True, **filters):
        """
        Download data products.
        If cloud access is enabled, files will be downloaded from the cloud if possible.

        Parameters
        ----------
        products : str, list, `~astropy.table.Table`
            Either a single or list of dataset IDs (as can be given to `get_product_list`),
            or a Table of products (as is returned by `get_product_list`)
        download_dir : str, optional
            Optional.  Directory to download files to.  Defaults to current directory.
        flat : bool, optional
            Default is False.  If set to True, and download_dir is specified, it will put
            all files into download_dir without subdirectories.  Or if set to True and
            download_dir is not specified, it will put files in the current directory,
            again with no subdirs.  The default of False puts files into the standard
            directory structure of "mastDownload/<obs_collection>/<obs_id>/".  If
            curl_flag=True, the flat flag has no effect, as astroquery does not control
            how MAST generates the curl download script.
        cache : bool, optional
            Default is True. If file is found on disc it will not be downloaded again.
            Note: has no affect when downloading curl script.
        curl_flag : bool, optional
            Default is False.  If true instead of downloading files directly, a curl script
            will be downloaded that can be used to download the data files at a later time.
        extension : string or array, optional
            Default None. Option to filter by file extension.
        verbose : bool, optional
            Default True. Whether to show download progress in the console.
        **filters :
            Filters to be applied.  Valid filters are all products fields returned by
            ``get_metadata("products")`` and 'extension' which is the desired file extension.
            The Column Name (or 'extension') is the keyword, with the argument being one or
            more acceptable values for that parameter.
            Filter behavior is AND between the filters and OR within a filter set.
            For example: productType="SCIENCE",extension=["fits","jpg"]

        Returns
        -------
        response : `~astropy.table.Table`
            The manifest of files downloaded, or status of files on disk if curl option chosen.
        """
        # If the products list is a row we need to cast it as a table
        if isinstance(products, Row):
            products = Table(products, masked=True)

        # If the products list is not already a table of products we need to
        # get the products and filter them appropriately
        if not isinstance(products, Table):

            if isinstance(products, str):
                products = [products]

            # collect list of products
            product_lists = []
            for oid in products:
                product_lists.append(self.get_product_list(oid))

            products = np.vstack(product_lists)

        # apply filters
        products = self.filter_products(products, extension=extension, **filters)

        # remove duplicate products
        products = utils.remove_duplicate_products(products, 'uri')

        if not len(products):
            warnings.warn("No products to download.", NoResultsWarning)
            return

        # set up the download directory and paths
        download_dir = '.' if not download_dir else download_dir

        if flat:
            base_dir = download_dir
        else:
            base_dir = Path(download_dir, "mastDownload")
        manifest = self._download_files(products,
                                        base_dir=base_dir, flat=flat,
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

        url = f"{conf.server}/search/util/api/v0.1/column_list?mission={self.mission}"

        try:
            results = requests.get(url)
            results = results.json()
            rows = []
            for result in results:
                result.pop('field_name')
                result.pop('queryable')
                result.pop('indexed')
                result.pop('default_output')
                rows.append((result['column_name'], result['qual_type'], result['description']))
            data_table = Table(rows=rows, names=('name', 'data_type', 'description'))
            return data_table
        except Exception:
            raise Exception(f"Error occurred while trying to get column list for mission {self.mission}")


MastMissions = MastMissionsClass()
