import numpy as np
from aiaccel.optimizer.abstract_optimizer import AbstractOptimizer
import random
from quasimc.sobol import Sobol
from decimal import Decimal, ROUND_HALF_UP

ORIGINAL_NUM = 10
NUM_PARENT = 5
EXPLORER_NUM = 70
SCHEDULE_DIV = 15


class MyOptimizer(AbstractOptimizer):
    """An optimizer class with a random algorithm."""

    def __init__(self, options: dict) -> None:
        super().__init__(options)
        self.parameter_pool = {}
        self.sobol = Sobol(len(self.params.get_parameter_list()))
        self.sobol_array = self.sobol.generate(ORIGINAL_NUM)
        # --- ランキング方式
        self.p_array_s = self.softmax(np.arange(NUM_PARENT, 0, -1))

    def _get_all_trial_id(self) -> list:
        """_get_all_trial_id.
        Get trial_ids from DB: 'result', 'finished', 'running', 'ready'
        Returns:
            List: trial_id
        """

        trial_id = self.storage.trial.get_all_trial_id()
        if trial_id is None:
            return []

        return trial_id

    def generate_parameter(self) -> None:
        """Generate parameters.

        Args:
            number (Optional[int]): A number of generating parameters.

        Returns:
            List[Dict[str, Union[str, float, List[float]]]]: A created
            list of parameters.
        """
        new_params = []
        hp_list = self.params.get_parameter_list()
        trial_id = self.trial_id.get()

        if trial_id <= ORIGINAL_NUM:
            new_params = self.get_original_params(trial_id, hp_list)
        else:
            new_params = self.get_next_params(trial_id, hp_list)

        self.parameter_pool[trial_id] = new_params

        return new_params

    def get_original_params(self, trial_id, hp_list):
        new_params = []
        for i, hp in enumerate(hp_list):
            sobol_value = self.sobol_array[i][trial_id - 1]
            # 対数スケールの場合
            if (hp.lower == 0 and hp.upper == 1) or (
                hp.lower / hp.upper <= 0.01 and hp.lower > 0
            ):
                if hp.lower == 0 and hp.upper == 1:
                    lower_index = min(-5, np.log10(hp.initial))
                else:
                    lower_index = np.log10(hp.lower)
                upper_index = np.log10(hp.upper)

                index_value = lower_index + (upper_index - lower_index) * sobol_value
                value = 10 ** index_value
            else:
                value = hp.lower + (hp.upper - hp.lower) * sobol_value
                if "INT" in hp.type:
                    value = int(
                        Decimal(str(value)).quantize(
                            Decimal("0"), rounding=ROUND_HALF_UP
                        )
                    )  # 四捨五入

            new_param = {"parameter_name": hp.name, "type": hp.type, "value": value}
            new_params.append(new_param)

        return new_params

    def get_next_params(self, trial_id, hp_list):
        trial_id_list = np.arange(1, len(self.parameter_pool) + 1, 1).tolist()

        objective_list = [
            self.storage.result.get_any_trial_objective(idx) for idx in trial_id_list
        ]
        objective_list = [i for i in objective_list if i is not None]

        trial_id_list = trial_id_list[: len(objective_list)]
        id_list_s = np.array(objective_list).argsort().tolist()
        trial_id_list_s = [i + 1 for i in id_list_s]
        objective_list_s = sorted(objective_list)

        trial_id_list_s = trial_id_list_s[:NUM_PARENT]
        objective_list_s = objective_list_s[:NUM_PARENT]

        next_params = []
        for j, hp in enumerate(hp_list):
            np.random.seed()
            t = np.random.choice(trial_id_list_s, p=self.p_array_s)
            hp_bests = [self.parameter_pool[x][j]["value"] for x in trial_id_list_s]
            next_param_value = self.parameter_pool[t][j]["value"]
            exploitation_rate = self.schedule_rate(trial_id)
            random.seed()
            if random.random() < exploitation_rate:
                change_value = self.get_change_value(trial_id, hp)
                next_param_value += change_value
                next_param_value = min(max(next_param_value, hp.lower), hp.upper)

            else:
                if self.trial_id.get() < EXPLORER_NUM:
                    next_param_value = random.uniform(hp.lower, hp.upper)
                    next_param_value = self.get_mutation_value(hp)
                else:
                    np.random.seed()
                    next_param_value = np.random.choice(hp_bests)

            next_param = {
                "parameter_name": hp.name,
                "type": hp.type,
                "value": next_param_value,
            }
            next_params.append(next_param)

        return next_params

    def get_change_value(self, trial_id, hp):
        np.random.seed()
        change_value = np.random.normal(0, (hp.upper - hp.lower) / trial_id)
        return change_value

    def get_mutation_value(self, hp):
        random.seed()
        new_value = random.uniform(hp.lower, hp.upper)
        return new_value

    def schedule_rate(self, trial_id):
        exploitation_rate = 1 / (
            1 + np.exp(-(trial_id - ORIGINAL_NUM - 1) / SCHEDULE_DIV)
        )  # sigmoidスケジューラ trial_id=ORIGINAL_NUM+1で0.5
        return exploitation_rate

    def softmax(self, value_list):
        exp_array = np.exp(value_list)
        softmax_array = exp_array / np.sum(exp_array)
        return softmax_array
