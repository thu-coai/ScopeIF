# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from verl.utils.import_utils import deprecated


def default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **kwargs,
):
    """Compute the score for a given solution based on the data source.

    The reward is selected purely by the `data_source` column of the training
    parquet, so the launch scripts differ only in the training file they read.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    if data_source == "scopeif":
        from . import scopeif

        res = scopeif.compute_score(solution_str, ground_truth, extra_info)
    elif data_source == "scopeif_wo_hra":
        from . import scopeif_wo_hra

        res = scopeif_wo_hra.compute_score(solution_str, ground_truth, extra_info)
    elif data_source == "rl_ila":
        from . import rl_ila

        res = rl_ila.compute_score(solution_str, ground_truth, extra_info)
    elif data_source == "rl_cla":
        from . import rl_cla

        res = rl_cla.compute_score(solution_str, ground_truth, extra_info)
    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, int | float | bool):
        return float(res)
    else:
        return float(res[0])


def default_compute_score_image(
    data_source,
    solution_image,
    ground_truth,
    extra_info=None,
    **kwargs,
):
    """Image-reward entry point. Unused by ScopeIF; kept so that
    verl.experimental.reward_loop.reward_manager.visual still imports."""
    raise NotImplementedError(f"Reward function is not implemented for {data_source=}")


@deprecated("verl.utils.reward_score.default_compute_score")
def _default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """
    Legacy function API to be deprecated. Please use `default_compute_score` instead.
    """
    return default_compute_score(data_source, solution_str, ground_truth, extra_info)


def get_default_compute_score(reward_name: str | None):
    """Get the default compute_score function based on the reward manager type."""
    if reward_name == "visual":
        return default_compute_score_image
    else:
        return default_compute_score


__all__ = ["default_compute_score"]
