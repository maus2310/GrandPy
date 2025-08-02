import re

import numpy as np
import pytest

from grandpy import read_grand


# Reading in the test data
@pytest.fixture(scope="module")
def nlls_ntr_dataset():
    sars = read_grand("../data/sars_R.tsv", design=("Condition", "dur.4sU", "Replicate"))
    return sars.normalize()

@pytest.fixture(scope="module")
def chase_dataset():
    chase = read_grand("https://zenodo.org/record/7612564/files/chase_notrescued.tsv.gz",
                       design=("Condition", "dur.4sU", "Replicate"),
                       rename_sample=lambda v: re.sub(r"\.chase", "", re.sub(r"0\.5h", "0_5h", re.sub(r"nos4U","no4sU",v))))
    chase = chase.filter_genes()
    return chase.normalize()


# Expected results extracted from GrandR
def expected_values_nlls_steady():
    return {
        "UHMK1": {
            "Mock_Synthesis": 175.36782,
            "Mock_Degradation": 0.09233363,
            "Mock_Half-life": 7.506985,
            "Mock_log_likelihood": -83.08113,
            "Mock_f0": 1899.28437,
            "Mock_total": 11012.2262,
            "Mock_conf_lower_Synthesis": 55.74426,
            "Mock_conf_lower_Degradation": 0.03680332,
            "Mock_conf_lower_Half-life": 4.6877366,
            "Mock_conf_upper_Synthesis": 294.99137,
            "Mock_conf_upper_Degradation": 0.1478639,
            "Mock_conf_upper_Half-life": 18.833820,
            "Mock_rmse": 245.792895,
            "Mock_rmse_new": 81.003241,
            "Mock_rmse_old": 338.033682,
            "SARS_Synthesis": 312.1146,
            "SARS_Degradation": 0.2463310,
            "SARS_Half-life": 2.8138849,
            "SARS_log_likelihood": -72.54345,
            "SARS_f0": 1267.0536,
            "SARS_total": 7586.693,
            "SARS_conf_lower_Synthesis": 239.60025,
            "SARS_conf_lower_Degradation": 0.1966457,
            "SARS_conf_lower_Half-life": 2.3415842,
            "SARS_conf_upper_Synthesis": 384.6290,
            "SARS_conf_upper_Degradation": 0.2960163,
            "SARS_conf_upper_Half-life": 3.524852,
            "SARS_rmse": 102.1405,
            "SARS_rmse_new": 103.4354,
            "SARS_rmse_old": 100.8289,
        },
        "ATF3": {
            "Mock_Synthesis": 33.97185,
            "Mock_Degradation": 0.73403905,
            "Mock_Half-life": 0.944292,
            "Mock_log_likelihood": -41.18428,
            "Mock_f0": 46.28071,
            "Mock_total": 303.7681,
            "Mock_conf_lower_Synthesis": 19.55973,
            "Mock_conf_lower_Degradation": 0.40194290,
            "Mock_conf_lower_Half-life": 0.6501494,
            "Mock_conf_upper_Synthesis": 48.38397,
            "Mock_conf_upper_Degradation": 1.0661352,
            "Mock_conf_upper_Half-life": 1.724492,
            "Mock_rmse": 7.486378,
            "Mock_rmse_new": 9.574865,
            "Mock_rmse_old": 4.518147,
            "SARS_Synthesis": 484.2662,
            "SARS_Degradation": 0.7425749,
            "SARS_Half-life": 0.9334374,
            "SARS_log_likelihood": -82.81963,
            "SARS_f0": 652.1446,
            "SARS_total": 3249.198,
            "SARS_conf_lower_Synthesis": 13.67737,
            "SARS_conf_lower_Degradation": 0.0,
            "SARS_conf_lower_Half-life": 0.4580502,
            "SARS_conf_upper_Synthesis": 954.8550,
            "SARS_conf_upper_Degradation": 1.5132560,
            "SARS_conf_upper_Half-life": np.nan,
            "SARS_rmse": 240.4945,
            "SARS_rmse_new": 146.8354,
            "SARS_rmse_old": 306.7810,
        },
        "PABPC4": {
            "Mock_Synthesis": 213.42534,
            "Mock_Degradation": 0.10611061,
            "Mock_Half-life": 6.532308,
            "Mock_log_likelihood": -69.18661,
            "Mock_f0": 2011.34770,
            "Mock_total": 12167.8269,
            "Mock_conf_lower_Synthesis": 174.41556,
            "Mock_conf_lower_Degradation": 0.08917529,
            "Mock_conf_lower_Half-life": 5.6332392,
            "Mock_conf_upper_Synthesis": 252.43512,
            "Mock_conf_upper_Degradation": 0.1230459,
            "Mock_conf_upper_Half-life": 7.772861,
            "Mock_rmse": 77.216403,
            "Mock_rmse_new": 42.920482,
            "Mock_rmse_old": 100.412041,
            "SARS_Synthesis": 525.5061,
            "SARS_Degradation": 0.2948892,
            "SARS_Half-life": 2.3505343,
            "SARS_log_likelihood": -78.40844,
            "SARS_f0": 1782.0458,
            "SARS_total": 10883.215,
            "SARS_conf_lower_Synthesis": 393.92966,
            "SARS_conf_lower_Degradation": 0.2293712,
            "SARS_conf_lower_Half-life": 1.9232332,
            "SARS_conf_upper_Synthesis": 657.0825,
            "SARS_conf_upper_Degradation": 0.3604072,
            "SARS_conf_upper_Half-life": 3.021946,
            "SARS_rmse": 166.5172,
            "SARS_rmse_new": 143.0915,
            "SARS_rmse_old": 187.0314,
        }
    }

def expected_values_nlls_nonsteady():
    return {
        "UHMK1": {
            "Mock_Synthesis": 159.10107,
            "Mock_Degradation": 0.15781737,
            "Mock_Half-life": 4.3920841,
            "Mock_log_likelihood": -81.96983,
            "Mock_f0": 2133.13999,
            "Mock_total": 11012.2262,
            "Mock_conf_lower_Synthesis": 29.43741,
            "Mock_conf_lower_Degradation": 0.03927958,
            "Mock_conf_lower_Half-life": 2.5081753,
            "Mock_conf_upper_Synthesis": 288.76473,
            "Mock_conf_upper_Degradation": 0.2763552,
            "Mock_conf_upper_Half-life": 17.646500,
            "Mock_rmse": 224.052544,
            "Mock_rmse_new": 56.414799,
            "Mock_rmse_old": 311.795535,
            "SARS_Synthesis": 309.9754,
            "SARS_Degradation": 0.2273713,
            "SARS_Half-life": 3.0485254,
            "SARS_log_likelihood": -72.40196,
            "SARS_f0": 1229.246,
            "SARS_total": 7586.693,
            "SARS_conf_lower_Synthesis": 233.9845,
            "SARS_conf_lower_Degradation": 0.1228627,
            "SARS_conf_lower_Half-life": 2.0885483,
            "SARS_conf_upper_Synthesis": 385.9663,
            "SARS_conf_upper_Degradation": 0.3318799,
            "SARS_conf_upper_Half-life": 5.641640,
            "SARS_rmse": 100.9432,
            "SARS_rmse_new": 102.2505,
            "SARS_rmse_old": 99.61872,
        },
        "ATF3": {
            "Mock_Synthesis": 34.55157,
            "Mock_Degradation": 0.76002299,
            "Mock_Half-life": 0.9120082,
            "Mock_log_likelihood": -41.16515,
            "Mock_f0": 47.39097,
            "Mock_total": 303.7681,
            "Mock_conf_lower_Synthesis": 17.39079,
            "Mock_conf_lower_Degradation": 0.26759172,
            "Mock_conf_lower_Half-life": 0.5534311,
            "Mock_conf_upper_Synthesis": 51.71235,
            "Mock_conf_upper_Degradation": 1.2524543,
            "Mock_conf_upper_Half-life": 2.590316,
            "Mock_rmse": 7.474453,
            "Mock_rmse_new": 9.415837,
            "Mock_rmse_old": 4.803845,
            "SARS_Synthesis": 992.6315,
            "SARS_Degradation": 2.5333532,
            "SARS_Half-life": 0.2736086,
            "SARS_log_likelihood": -73.68256,
            "SARS_f0": 1307.664,
            "SARS_total": 3249.198,
            "SARS_conf_lower_Synthesis": 0, # R: -59.6934
            "SARS_conf_lower_Degradation": 0, # R: -0.1503619
            "SARS_conf_lower_Half-life": 0.1328614,
            "SARS_conf_upper_Synthesis": 2044.9563,
            "SARS_conf_upper_Degradation": 5.2170684,
            "SARS_conf_upper_Half-life": np.nan, # R: -4.609860
            "SARS_rmse": 112.3114,
            "SARS_rmse_new": 155.9911,
            "SARS_rmse_old": 29.90762,
        },
        "PABPC4": {
            "Mock_Synthesis": 216.87770,
            "Mock_Degradation": 0.09028628,
            "Mock_Half-life": 7.6772151,
            "Mock_log_likelihood": -68.39860,
            "Mock_f0": 1953.78231,
            "Mock_total": 12167.8269,
            "Mock_conf_lower_Synthesis": 177.89985,
            "Mock_conf_lower_Degradation": 0.05382914,
            "Mock_conf_lower_Half-life": 5.4689011,
            "Mock_conf_upper_Synthesis": 255.85556,
            "Mock_conf_upper_Degradation": 0.1267434,
            "Mock_conf_upper_Half-life": 12.876801,
            "Mock_rmse": 72.308720,
            "Mock_rmse_new": 40.521987,
            "Mock_rmse_old": 93.888607,
            "SARS_Synthesis": 516.2158,
            "SARS_Degradation": 0.2553116,
            "SARS_Half-life": 2.7149065,
            "SARS_log_likelihood": -77.94075,
            "SARS_f0": 1675.183,
            "SARS_total": 10883.215,
            "SARS_conf_lower_Synthesis": 382.4891,
            "SARS_conf_lower_Degradation": 0.1278815,
            "SARS_conf_lower_Half-life": 1.8110047,
            "SARS_conf_upper_Synthesis": 649.9425,
            "SARS_conf_upper_Degradation": 0.3827418,
            "SARS_conf_upper_Half-life": 5.420232,
            "SARS_rmse": 160.1521,
            "SARS_rmse_new": 146.7128,
            "SARS_rmse_old": 172.54779,
        }
    }

def expected_values_ntr():
    return {
        "UHMK1": {
            "Mock_Synthesis": 146.3304,
            "Mock_Degradation": 0.0792638,
            "Mock_Half-life": 8.744814,
            "Mock_log_likelihood": -7782.4123,
            "Mock_f0": 1846.11865,
            "Mock_total": 1846.11865,
            "Mock_conf_lower_Synthesis": 140.06364,
            "Mock_conf_lower_Degradation": 0.07586925,
            "Mock_conf_lower_Half-life": 8.3722176,
            "Mock_conf_upper_Synthesis": 152.84265,
            "Mock_conf_upper_Degradation": 0.08279135,
            "Mock_conf_upper_Half-life": 9.136075,
            "Mock_rmse": 0.01550568,
            "SARS_Synthesis": 315.7082,
            "SARS_Degradation": 0.2486066,
            "SARS_Half-life": 2.7881289,
            "SARS_log_likelihood": -1648.47502,
            "SARS_f0": 1269.911,
            "SARS_total": 1269.911,
            "SARS_conf_lower_Synthesis": 292.1833,
            "SARS_conf_lower_Degradation": 0.2300818,
            "SARS_conf_lower_Half-life": 2.5853129,
            "SARS_conf_upper_Synthesis": 340.4753,
            "SARS_conf_upper_Degradation": 0.2681096,
            "SARS_conf_upper_Half-life": 3.0126126,
            "SARS_rmse": 0.04563580,
        },
        "ATF3": {
            "Mock_Synthesis": 26.0133,
            "Mock_Degradation": 0.5784609,
            "Mock_Half-life": 1.198261,
            "Mock_log_likelihood": -142.1615,
            "Mock_f0": 44.96985,
            "Mock_total": 44.96985,
            "Mock_conf_lower_Synthesis": 21.09593,
            "Mock_conf_lower_Degradation": 0.46911273,
            "Mock_conf_lower_Half-life": 0.9806687,
            "Mock_conf_upper_Synthesis": 31.78517,
            "Mock_conf_upper_Degradation": 0.70681078,
            "Mock_conf_upper_Half-life": 1.477571,
            "Mock_rmse": 0.10109058,
            "SARS_Synthesis": 933.9800,
            "SARS_Degradation": 2.2299260,
            "SARS_Half-life": 0.3108386,
            "SARS_log_likelihood": -14.29713,
            "SARS_f0": 418.839,
            "SARS_total": 418.839,
            "SARS_conf_lower_Synthesis": 657.3969,
            "SARS_conf_lower_Degradation": 1.5695693,
            "SARS_conf_lower_Half-life": 0.2058735,
            "SARS_conf_upper_Synthesis": 1410.1722,
            "SARS_conf_upper_Degradation": 3.3668597,
            "SARS_conf_upper_Half-life": 0.4416162,
            "SARS_rmse": 0.02143381,
        },
        "PABPC4": {
            "Mock_Synthesis": 220.5477,
            "Mock_Degradation": 0.1089119,
            "Mock_Half-life": 6.364290,
            "Mock_log_likelihood": -8060.8233,
            "Mock_f0": 2025.00938,
            "Mock_total": 2025.00938,
            "Mock_conf_lower_Synthesis": 211.69266,
            "Mock_conf_lower_Degradation": 0.10453910,
            "Mock_conf_lower_Half-life": 6.1115503,
            "Mock_conf_upper_Synthesis": 229.66833,
            "Mock_conf_upper_Degradation": 0.11341593,
            "Mock_conf_upper_Half-life": 6.630506,
            "Mock_rmse": 0.01041070,
            "SARS_Synthesis": 580.6547,
            "SARS_Degradation": 0.3066879,
            "SARS_Half-life": 2.2601062,
            "SARS_log_likelihood": -1631.67109,
            "SARS_f0": 1893.308,
            "SARS_total": 1893.308,
            "SARS_conf_lower_Synthesis": 538.9911,
            "SARS_conf_lower_Degradation": 0.2846822,
            "SARS_conf_lower_Half-life": 2.1013362,
            "SARS_conf_upper_Synthesis": 624.5270,
            "SARS_conf_upper_Degradation": 0.3298602,
            "SARS_conf_upper_Half-life": 2.4348104,
            "SARS_rmse": 0.05942439,
        }
    }

def expected_values_ntr_exact_ci():
    return {
        "UHMK1": {
            "Mock_conf_lower_Synthesis": 141.32705,
            "Mock_conf_lower_Degradation": 0.07655361,
            "Mock_conf_lower_Half-life": 8.442349,
            "Mock_conf_upper_Synthesis": 151.57298,
            "Mock_conf_upper_Degradation": 0.08210359,
            "Mock_conf_upper_Half-life": 9.054402,
            "SARS_conf_lower_Synthesis": 296.9834,
            "SARS_conf_lower_Degradation": 0.2338616,
            "SARS_conf_lower_Half-life": 2.6225051,
            "SARS_conf_upper_Synthesis": 335.6467,
            "SARS_conf_upper_Degradation": 0.2643073,
            "SARS_conf_upper_Half-life": 2.9639205,
        },
        "ATF3": {
            "Mock_conf_lower_Synthesis": 22.13927,
            "Mock_conf_lower_Degradation": 0.49231355,
            "Mock_conf_lower_Half-life": 1.014637,
            "Mock_conf_upper_Synthesis": 30.72107,
            "Mock_conf_upper_Degradation": 0.68314818,
            "Mock_conf_upper_Half-life": 1.407938,
            "SARS_conf_lower_Synthesis": 722.8896,
            "SARS_conf_lower_Degradation": 1.7259367,
            "SARS_conf_lower_Half-life": 0.2156857,
            "SARS_conf_upper_Synthesis": 1346.0191,
            "SARS_conf_upper_Degradation": 3.2136908,
            "SARS_conf_upper_Half-life": 0.4016064,
        },
        "PABPC4": {
            "Mock_conf_lower_Synthesis": 213.50440,
            "Mock_conf_lower_Degradation": 0.10543379,
            "Mock_conf_lower_Half-life": 6.158838,
            "Mock_conf_upper_Synthesis": 227.90495,
            "Mock_conf_upper_Degradation": 0.11254513,
            "Mock_conf_upper_Half-life": 6.574242,
            "SARS_conf_lower_Synthesis": 547.4334,
            "SARS_conf_lower_Degradation": 0.2891412,
            "SARS_conf_lower_Half-life": 2.1305150,
            "SARS_conf_upper_Synthesis": 615.9737,
            "SARS_conf_upper_Degradation": 0.3253425,
            "SARS_conf_upper_Half-life": 2.3972618,
        }
    }

def expected_values_chase():
    return {
        "Qsox1.155778412": {
            "mESC_Synthesis": 27.46546,
            "mESC_Degradation": 0.20088766,
            "mESC_Half-life": 3.450422,
            "mESC_log_likelihood": 15.19151,
            "mESC_f0": 136.7205,
            "mESC_total": 136.7205,
            "mESC_conf_lower_Synthesis": 0.10020028,
            "mESC_conf_lower_Degradation": 0.12937742,
            "mESC_conf_lower_Half-life": 2.544613,
            "mESC_conf_upper_Synthesis": 0.25291023,
            "mESC_conf_upper_Degradation": 0.27239790,
            "mESC_conf_upper_Half-life": 5.357559,
            "mESC_rmse": 0.08299979,
        },
        "Ipo9.135384724": {
            "mESC_Synthesis": 42.07806,
            "mESC_Degradation": 0.11330271,
            "mESC_Half-life": 6.117658,
            "mESC_log_likelihood": 13.98979,
            "mESC_f0": 371.3774,
            "mESC_total": 371.3774,
            "mESC_conf_lower_Synthesis": 0.08041326,
            "mESC_conf_lower_Degradation": 0.08022190,
            "mESC_conf_lower_Half-life": 4.735145,
            "mESC_conf_upper_Synthesis": 0.16678404,
            "mESC_conf_upper_Degradation": 0.14638352,
            "mESC_conf_upper_Half-life": 8.640373,
            "mESC_rmse": 0.08788794,
        },
        "Rpl37a.72713813": {
            "mESC_Synthesis": 180.72132,
            "mESC_Degradation": 0.06434391,
            "mESC_Half-life": 10.772538,
            "mESC_log_likelihood": 31.30983,
            "mESC_f0": 2808.6781,
            "mESC_total": 2808.6781,
            "mESC_conf_lower_Synthesis": 0.03536533,
            "mESC_conf_lower_Degradation": 0.05164275,
            "mESC_conf_lower_Half-life": 8.996646,
            "mESC_conf_upper_Synthesis": 0.05670633,
            "mESC_conf_upper_Degradation": 0.07704506,
            "mESC_conf_upper_Half-life": 13.421964,
            "mESC_rmse": 0.03852467,
        }
    }


# Comparing results to expected values
def test_fit_kinetics_nlls_steady(nlls_ntr_dataset):
    nlls_result = nlls_ntr_dataset.fit_kinetics(
        fit_type="nlls",
        genes=["UHMK1", "ATF3", "PABPC4"],
        return_fields=[
            "Synthesis", "Degradation", "Half-life",
            "log_likelihood", "f0", "total",
            "conf_lower", "conf_upper", "rmse",
            "rmse_old", "rmse_new"
        ],
        prefix=None,
        show_progress=False,
        steady_state=True
    )

    table = nlls_result.get_analysis_table(with_gene_info=False)
    expected = expected_values_nlls_steady()

    for gene, fields in expected.items():
        for colname, exp_val in fields.items():
            actual = table.loc[gene, colname]

            assert np.isclose(actual, exp_val, rtol=1e-1, atol=1e-4, equal_nan=True), (
                f"{gene} {colname}: expected {exp_val}, got {actual}"
            )

def test_fit_kinetics_nlls_nonsteady(nlls_ntr_dataset):
    nlls_result = nlls_ntr_dataset.fit_kinetics(
        fit_type="nlls",
        genes=["UHMK1", "ATF3", "PABPC4"],
        return_fields=[
            "Synthesis", "Degradation", "Half-life",
            "log_likelihood", "f0", "total",
            "conf_lower", "conf_upper", "rmse",
            "rmse_old", "rmse_new"
        ],
        prefix=None,
        show_progress=False,
        steady_state=False
    )

    table = nlls_result.get_analysis_table(with_gene_info=False)
    expected = expected_values_nlls_nonsteady()

    for gene, fields in expected.items():
        for colname, exp_val in fields.items():
            actual = table.loc[gene, colname]
            assert np.isclose(actual, exp_val, rtol=1e-1, atol=1e-4, equal_nan=True), (
                f"{gene} {colname}: expected {exp_val}, got {actual}"
            )

def test_fit_kinetics_ntr(nlls_ntr_dataset):
    ntr_result = nlls_ntr_dataset.fit_kinetics(
        fit_type="ntr",
        genes=["UHMK1", "ATF3", "PABPC4"],
        return_fields=[
            "Synthesis", "Degradation", "Half-life",
            "log_likelihood", "f0", "total",
            "conf_lower", "conf_upper", "rmse"
        ],
        prefix=None,
        exact_ci=False,
        show_progress=False
    )

    table = ntr_result.get_analysis_table(with_gene_info=False)
    expected = expected_values_ntr()

    for gene, fields in expected.items():
        for colname, exp_val in fields.items():
            actual = table.loc[gene, colname]
            assert np.isclose(actual, exp_val, rtol=1e-2, atol=1e-6, equal_nan=True), (
                f"{gene} {colname}: expected {exp_val}, got {actual}"
            )

def test_fit_kinetics_ntr_exact_ci(nlls_ntr_dataset):
    ntr_result = nlls_ntr_dataset.fit_kinetics(
        fit_type="ntr",
        genes=["UHMK1", "ATF3", "PABPC4"],
        return_fields=["conf_lower", "conf_upper"],
        prefix=None,
        exact_ci=True,
        show_progress=False
    )

    table = ntr_result.get_analysis_table(with_gene_info=False)
    expected = expected_values_ntr_exact_ci()

    for gene, fields in expected.items():
        for colname, exp_val in fields.items():
            actual = table.loc[gene, colname]
            assert np.isclose(actual, exp_val, rtol=1e-2, atol=1e-6, equal_nan=True), (
                f"{gene} {colname}: expected {exp_val}, got {actual}"
            )

def test_fit_kinetics_chase(chase_dataset):
    chase_result = chase_dataset.fit_kinetics(
        fit_type="chase",
        genes=["Qsox1.155778412","Ipo9.135384724","Rpl37a.72713813"],
        return_fields=[
            "Synthesis", "Degradation", "Half-life",
            "log_likelihood", "f0", "total",
            "conf_lower", "conf_upper", "rmse"
        ],
        prefix=None,
        show_progress=False
    )

    table = chase_result.get_analysis_table(with_gene_info=False)
    expected = expected_values_chase()

    for gene, fields in expected.items():
        for colname, exp_val in fields.items():
            actual = table.loc[gene, colname]
            assert np.isclose(actual, exp_val, rtol=1, atol=1e-4, equal_nan=True), (
                f"{gene} {colname}: expected {exp_val}, got {actual}"
            )