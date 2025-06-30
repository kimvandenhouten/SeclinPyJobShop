import os
import time
import copy
import numpy as np

import general.logger
from pyjobshop.Model import Model, Solution

from PyJobShopIntegration.simulator import Simulator
from PyJobShopIntegration.utils import find_schedule_per_resource
from PyJobShopIntegration.Sampler import DiscreteRVSampler
from PyJobShopIntegration.PyJobShopSTNU import PyJobShopSTNU
from temporal_networks.cstnu_tool.stnu_to_xml_function import stnu_to_xml
from temporal_networks.cstnu_tool.call_java_cstnu_tool import run_dc_algorithm
from temporal_networks.rte_star import rte_star

logger = general.logger.get_logger(__name__)


def run_proactive_offline_pjstnu(
    model_factory,
    duration_distributions: DiscreteRVSampler,
    mode: str = "robust",
    quantile: float = 0.5,
    saa_samples: int = 10,
    dc_xml_folder: str = "temporal_networks/cstnu_tool/xml_files",
):
    """
    Build a \emph{proactive} STNU‐based schedule:
      - mode="robust": use upper‐bounds of durations
      - mode="quantile": use the p‐quantile of [lb,ub]
      - mode="SAA": will just build the STNU for later SAA online
    Returns a dict with:
      'stnu', 'solution', 'dc' (bool), 'offline_time', 'estimated_durations'
    """
    data = {
        "method": f"pjstnu_{mode}",
        "offline_time": None,
        "dc": False,
        "solution": None,
        "stnu": None,
        "estimated_durations": None,
    }

    # 1) pick an “estimated” duration vector
    lb, ub = duration_distributions.get_bounds()
    n = len(lb)
    if mode == "robust":
        est = ub.copy()
    elif mode.startswith("quantile"):
        est = np.floor(lb + quantile * (ub - lb)).astype(int)
    elif mode == "SAA":
        est = None  # we’ll sample in the online stage
    else:
        raise ValueError(f"Unknown mode {mode}")

    data["estimated_durations"] = est

    # 2) Solve the CP model with these estimated durations
    #    NOTE: PyJobShop’s Model doesn’t natively let you override the durations
    #    per mode. A quick hack: rebuild the model with only one mode/task
    #    whose duration = est[i], but keep the same machine assignment.
    t0 = time.time()
    model = model_factory()
    # if est is not None, you would here replace model.add_mode calls
    # to only add one mode with duration=est[task_idx].  For now we assume
    # your model_factory already builds with the “nominal” durations.
    res = model.solve(display=False)
    data["offline_time"] = time.time() - t0
    sol: Solution = res.best
    data["solution"] = sol

    # 3) Build the STNU from that solution + original distribution
    stnu = PyJobShopSTNU.from_concrete_model(model, duration_distributions)
    stnu.add_resource_chains(sol, model)
    data["stnu"] = stnu

    # 4) Check dynamic controllability
    os.makedirs(dc_xml_folder, exist_ok=True)
    stnu_to_xml(stnu, f"pjstnu_{mode}", dc_xml_folder)
    dc, _ = run_dc_algorithm(dc_xml_folder, f"pjstnu_{mode}")
    data["dc"] = dc

    return data


def run_proactive_online_pjstnu(
    offline_data: dict,
    duration_distributions: DiscreteRVSampler,
    n_runs: int = 100
):
    """
    Given the dict returned by run_proactive_offline_pjstnu, sample n_runs
    durations and dispatch them through RTE* on the STNU, recording feasibility
    and realized makespan for each sample.
    Returns a list of per‐run dicts:
      { 'feasible': bool, 'makespan': float, 'duration_sample': np.array }
    """
    if not offline_data["dc"]:
        logger.warning("STNU is not dynamically controllable → no online sim.")
        return None

        # now just use your Simulator you already have
    sim = Simulator(
        offline_data["model"],
        offline_data["stnu"],
        offline_data["solution"],
        duration_distributions,
        objective="makespan"
    )
    return sim.run_many(runs=n_runs)
