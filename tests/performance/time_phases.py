import pandas as pd

from pandas_profiling import ProfileReport, Loger


def data_sets():
    def california_housing():
        from sklearn.datasets import fetch_california_housing

        data_set = fetch_california_housing()

        return pd.DataFrame(data_set.data, columns=data_set.feature_names)

    def meteorites():
        return pd.read_csv("../../data/meteorites.csv")

    def titanic():
        return pd.read_csv("../../data/titanic.csv")

    def us_census_demographic_data_2017():
        """https://www.kaggle.com/muonneutrino/us-census-demographic-data"""
        return pd.read_csv("./data/acs2017_census_tract_data.csv")

    def us_census_demographic_data_2015():
        """https://www.kaggle.com/muonneutrino/us-census-demographic-data"""

        return pd.read_csv("./data/acs2015_census_tract_data.csv")

    def new_york_stock_exchange():
        """https://www.kaggle.com/dgawlik/nyse"""
        return pd.read_csv("./data/new_york_stock_exchange_prices.csv")

    def electric_power_consumption():
        """https://www.kaggle.com/uciml/electric-power-consumption-data-set"""
        return pd.read_csv("./data/household_power_consumption.csv", na_values=["?"])

    for get_df, name in [
        (titanic, "titanic"),
        (california_housing, "california_housing"),
        (meteorites, "meteorites"),
        (us_census_demographic_data_2015, "us_census_demographic_data_2015"),
        (us_census_demographic_data_2017, "us_census_demographic_data_2017"),
        (new_york_stock_exchange, "new_york_stock_exchange"),
        (electric_power_consumption, "electric_power_consumption"),
    ]:
        yield get_df(), name


def assert_description_equal(d1, d2):
    assert str(d1["table"]) == str(d2["table"])
    assert str(d1["variables"]) == str(d2["variables"])
    assert str(d1["correlations"]) == str(d2["correlations"])
    assert str(d1["messages"]) == str(d2["messages"])
    # assert str(d1["missing"]) == str(d2["missing"]) # clip-path is different
    # assert str(d1["package"]) == str(d2["package"]) # config may be different


def run_real_data():
    for df, name in data_sets():
        Loger.clear()
        print("=" * 79)
        print(name, df.shape)
        profile = ProfileReport(df)
        profile.to_html()
        print(f"Total time:{Loger.total_time()}")
        print(Loger.to_str(time_fliter=0.1, proportion=True))


if __name__ == "__main__":
    run_real_data()
