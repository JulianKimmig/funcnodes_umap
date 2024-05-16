import funcnodes as fn
import umap
from typing import Literal, Optional, Any, Callable, Union, Tuple
import numpy as np
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    RobustScaler,
)


@fn.NodeDecorator(
    node_id="umap.reducer",
    node_name="UMAP Reducer",
    default_io_options={
        "n_components": {"value_options": {"min": 2, "max": 100}},
        "n_neighbors": {"value_options": {"min": 2}},
    },
    outputs=[{"name": "reducer_gen"}],
)
def reducer(
    n_neighbors: int = 15,
    n_components: int = 2,
    min_dist: float = 0.1,
    spread: float = 1.0,
    metric: Literal[
        "euclidean",
        "manhattan",
        "chebyshev",
        "minkowski",
        "canberra",
        "braycurtis",
        "mahalanobis",
        "wminkowski",
        "seuclidean",
        "cosine",
        "correlation",
        "haversine",
        "hamming",
        "jaccard",
        "dice",
        "russelrao",
        "kulsinski",
        "ll_dirichlet",
        "hellinger",
        "rogerstanimoto",
        "sokalmichener",
        "sokalsneath",
        "yule",
    ] = "euclidean",
    n_epochs: Optional[int] = None,
    learning_rate: float = 1.0,
    init: Literal["spectral", "random", "pca", "tswspectral"] = "spectral",
    random_state: int = 42,
) -> umap.UMAP:
    def make_reducer():
        return umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            random_state=random_state,
            init=init,
            spread=spread,
        )

    return make_reducer


@fn.NodeDecorator(
    node_id="umap.fit",
    node_name="UMAP Fit Transform",
    outputs=[{"name": "embedding"}, {"name": "reducer"}],
)
def umap_fit_transform(
    reducer: Union[umap.UMAP, Callable[[], umap.UMAP]],
    data: Any,
    scaler: Union[
        Literal[
            "None",
            "StandardScaler",
            "MinMaxScaler",
            "MaxAbsScaler",
            "RobustScaler",
        ],
        StandardScaler,
        MinMaxScaler,
        MaxAbsScaler,
        RobustScaler,
    ] = "StandardScaler",
) -> Tuple[
    np.ndarray,
    umap.UMAP,
    Union[StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, None],
]:
    if not isinstance(reducer, umap.UMAP):
        reducer = reducer()

    if isinstance(scaler, str):
        if scaler == "StandardScaler":
            innerscaler = StandardScaler()
        elif scaler == "MinMaxScaler":
            innerscaler = MinMaxScaler()
        elif scaler == "MaxAbsScaler":
            innerscaler = MaxAbsScaler()
        elif scaler == "RobustScaler":
            innerscaler = RobustScaler()
        elif scaler == "None":
            innerscaler = None
        else:
            raise ValueError(f"Scaler {scaler} not supported")
    else:
        innerscaler = scaler

    if innerscaler:
        data = innerscaler.fit_transform(data)
    embedding = reducer.fit_transform(data)

    return embedding, reducer, innerscaler
