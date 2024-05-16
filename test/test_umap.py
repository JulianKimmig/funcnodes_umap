import funcnodes as fn
import funcnodes_umap as fnumap
import unittest
import funcnodes_pandas as fndp
import pandas as pd
import numpy as np

import os
import matplotlib.pyplot as plt

df = pd.read_csv(
    os.path.join(
        os.path.dirname(__file__),
        "penguins.csv",
    )
)


class TestUMAP(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.df = df

    async def test_umap(self):
        dropnan = fndp.dropna()
        dropnan.inputs["df"].value = self.df

        to_numeric = fndp.numeric_only()
        to_numeric.inputs["df"].connect(dropnan.outputs["out"])
        to_numeric.inputs["label_encode"].value = True

        dropcols = fndp.drop_columns()
        dropcols.inputs["df"].connect(dropnan.outputs["out"])
        dropcols.inputs["columns"].value = "species, island, sex, year"

        await fn.run_until_complete(to_numeric, dropnan, dropcols)
        df = dropcols.outputs["out"].value
        self.assertIsInstance(df, pd.DataFrame)

        reducer = fnumap.reducer()

        fit = fnumap.umap_fit_transform()
        fit.inputs["reducer"].connect(reducer.outputs["reducer_gen"])
        fit.inputs["data"].connect(dropcols.outputs["out"])

        await fn.run_until_complete(fit, reducer, dropcols)

        self.assertEqual(
            fit.inputs["data"].value.columns.tolist(),
            [
                "bill_length_mm",
                "bill_depth_mm",
                "flipper_length_mm",
                "body_mass_g",
            ],
        )

        embedding = fit.outputs["embedding"].value
        # embedding = np.random.rand(df.shape[0] * 2).reshape((df.shape[0], 2))
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.shape, (df.shape[0], 2))

        add_column = fndp.add_column()
        add_column.inputs["df"].connect(dropnan.outputs["out"])
        add_column.inputs["data"].value = [r for r in embedding]
        add_column.inputs["column"].value = "embedding"

        groupby = fndp.group_by()
        groupby.inputs["df"].connect(add_column.outputs["out"])
        groupby.inputs["by"].value = "species"

        await fn.run_until_complete(add_column, groupby, dropnan)

        get_g = fndp.get_df_from_group()
        get_g.inputs["group"].connect(groupby.outputs["grouped"])
        await get_g
        for o in get_g.inputs["name"].value_options["options"]:
            get_g.inputs["name"].value = o
            await get_g

            df = get_g.outputs["df"].value
            emb = np.array([r.tolist() for r in df["embedding"]])
            plt.scatter(
                emb[:, 0],
                emb[:, 1],
                label=o,
            )

        plt.legend()
        plt.show()
