# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Catalog-bound query helpers and column references for MAST catalogs."""

from dataclasses import dataclass, field

from ..exceptions import InvalidQueryError
from .collections import CatalogsClass

__all__ = ["Catalog", "CatalogColumn"]


@dataclass(frozen=True)
class CatalogColumn:
    """An immutable, hashable column reference belonging to a `Catalog`."""

    catalog: "Catalog"
    name: str
    description: str | None = field(default=None, compare=False, repr=False)

    def __repr__(self):
        label = f"{self.catalog.catalog_name}.{self.name}"
        return f"{label} ({self.description})" if self.description else label


class Catalog:
    """
    A MAST catalog with column references for server-side crossmatching.

    Parameters
    ----------
    collection : str
        MAST collection name.
    catalog : str, optional
        Catalog name, with or without its schema prefix. If omitted, use the
        collection's default catalog.

    Notes
    -----
    Construction validates the collection and catalog and may fetch discovery
    metadata. Column metadata is fetched on first column access and cached by
    the existing collection API.
    """

    def __init__(self, collection, catalog=None):
        if not isinstance(collection, str) or not collection.strip():
            raise InvalidQueryError("Collection name must be a non-empty string.")
        if catalog is not None and (not isinstance(catalog, str) or not catalog.strip()):
            raise InvalidQueryError("Catalog name must be a non-empty string.")
        self._query = CatalogsClass(collection=collection, catalog=catalog)

    @property
    def collection_name(self):
        """The resolved MAST collection name."""
        return self._query.collection

    @property
    def catalog_name(self):
        """The resolved catalog name, including its schema prefix."""
        return self._query.catalog

    @property
    def columns(self):
        """Column metadata as an `~astropy.table.Table`."""
        return self._query.get_column_metadata()

    def get_crossmatch_catalogs(self):
        """
        List same-endpoint catalog candidates with identifiable RA/Dec columns.

        Returns
        -------
        Table
            An `~astropy.table.Table` listing the crossmatchable catalogs.
        """
        return self._query.get_crossmatch_catalogs(
            collection=self.collection_name,
            catalog=self.catalog_name,
        )

    def __repr__(self):
        """String representation of the Catalog object."""
        return f"Catalog(collection={self.collection_name!r}, catalog={self.catalog_name!r})"

    def __getitem__(self, names):
        """
        Return a column reference, or a tuple of references for multiple names.

        Indexing selects query columns, not rows or downloaded column data.
        Names are matched case-insensitively against the catalog metadata.
        """
        if isinstance(names, (tuple, list)):
            return tuple(self[name] for name in names)
        if not isinstance(names, str) or not names.strip():
            raise InvalidQueryError("Column names must be non-empty strings.")
        columns = self.columns
        for index, name in enumerate(columns["column_name"]):
            if name.casefold() == names.strip().casefold():
                description = None
                if "description" in columns.colnames:
                    value = str(columns["description"][index]).strip()
                    description = value if value and value != "--" else None
                return CatalogColumn(self, str(name), description)
        raise KeyError(f"Column '{names}' was not found in catalog '{self.catalog_name}'.")

    def __getattr__(self, name):
        """Allow access to columns as attributes."""
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(str(exc)) from exc

    def _check_query_target(self, kwargs):
        """Check that query target parameters are not overridden."""
        if {"collection", "catalog"}.intersection(kwargs):
            raise InvalidQueryError("Collection and catalog are fixed by this Catalog object.")

    def query_criteria(self, **kwargs):
        """
        Query this catalog by criteria, optionally including a spatial constraint.

        Parameters
        ----------
        **kwargs
            Keyword arguments specifying the query criteria and optional spatial constraints.

        Returns
        -------
        astropy.table.Table
            The result of the query.
        """
        self._check_query_target(kwargs)
        return self._query.query_criteria(**kwargs)

    def query_region(self, coordinates=None, *, region=None, **kwargs):
        """
        Query this catalog within a sky region or around coordinates.

        Parameters
        ----------
        coordinates : astropy.coordinates.SkyCoord, optional
            The sky coordinates to query around.
        region : str, optional
            The sky region to query within.
        **kwargs
            Additional keyword arguments passed to the underlying query method.

        Returns
        -------
        astropy.table.Table
            The result of the query.
        """
        self._check_query_target(kwargs)
        return self._query.query_region(coordinates, region=region, **kwargs)

    def query_object(self, object_name, **kwargs):
        """
        Query this catalog around a named astronomical object.

        Parameters
        ----------
        object_name : str
            The name of the astronomical object to query around.
        **kwargs
            Additional keyword arguments passed to the underlying query method.

        Returns
        -------
        astropy.table.Table
            The result of the query.
        """
        self._check_query_target(kwargs)
        return self._query.query_object(object_name, **kwargs)

    def crossmatch(self, other, *, select_cols=None, filters=None, sort_by=None, **kwargs):
        """
        Crossmatch this catalog with another `Catalog`.

        ``select_cols`` accepts strings, `CatalogColumn` references, and groups
        returned by indexing multiple columns. ``filters`` accepts column
        references as dictionary keys, and ``sort_by`` accepts references too.
        References must belong to one of the two participating catalog objects.

        Other parameters and return values follow
        `~astroquery.mast.CatalogsClass.crossmatch`, including spatial constraints,
        aliases, keyword criteria, ``count_only``, and ``return_adql``.
        Unqualified strings and keyword criteria retain the existing ambiguity
        checks; use column references to disambiguate shared column names.

        Parameters
        ----------
        other : Catalog
            The other catalog to crossmatch with.
        select_cols : list or str, optional
            Columns to include in the result.
        filters : dict, optional
            Filters to apply to the crossmatch.
        sort_by : list or str, optional
            Columns to sort the result by.
        **kwargs
            Additional keyword arguments passed to the underlying crossmatch method.

        Returns
        -------
        astropy.table.Table
            The result of the crossmatch query.
        """
        if not isinstance(other, Catalog):
            raise InvalidQueryError("Crossmatch requires another Catalog object.")
        reserved = {"catalogs", "targets", "collection_1", "collection_2", "catalog_1", "catalog_2",
                    "collection1", "collection2", "catalog1", "catalog2"}
        if reserved.intersection(kwargs):
            raise InvalidQueryError("Crossmatch catalogs are supplied by the two Catalog objects.")

        return self._query.crossmatch(
            catalogs=(self, other),
            select_cols=select_cols, filters=filters, sort_by=sort_by, **kwargs,
        )
