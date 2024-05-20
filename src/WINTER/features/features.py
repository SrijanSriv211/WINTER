from ..shared.utils import get_bin_path, dprint
from . import func
import random, json, os

class Features:
    def __init__(self, filepath):
        self.filepath = filepath

        with open(self.filepath, "r", encoding="utf-8") as f:
            self.jsondata = json.load(f)
        self.func_dict = {}

    # Map all the functions corresponding to their respective skills.
    def load(self):
        self.func_dict = {
            "play,video": func.play_video,
            "play,music": func.play_music,
            "open,game": func.playgames,
            "play,on_youtube": func.youtube,
            "search,wikipedia": func.search_on_wikipedia,
            "translate,one_lang_to_another": func.translate
        }

    # There are 4 execution engines: AOs, func, skills and external.
    def execute(self, input_prompt, predicted_output, respond=True):
        skillname = predicted_output[0].split(";")[1]
        score = predicted_output[1]

        if score < 0.6:
            # The "default" skill states that the particular input is actually a conversation rather than a intent.
            skillname = "default"
            score = 1

        for intent in self.jsondata["skills"]:
            if skillname == intent["skill"]:
                tasks = intent["tasks"]
                responses = intent["responses"]
                response_config = intent["response_config"]
                break

        if respond:
            intent["response_config"] = self.__give_response__(responses, response_config)
            self.__update_response_config__()

        self.__exec_tasks__(input_prompt, tasks, skillname)

    def __update_response_config__(self):
        # Serializing json
        json_obj = json.dumps(self.jsondata, indent=4)
        
        # Writing to sample.json
        with open(self.filepath, "w", encoding="utf-8") as f:
            f.write(json_obj)

    def __give_response__(self, responses, response_config):
        enable_dprint = response_config["dprint"]
        shuffle = response_config["shuffle"]
        do_reverse = response_config["reverse"]
        shuffle_seed = response_config["shuffle_seed"]

        # The response won't be selected randomly and, responses will be printed sequentially and only once.
        # Responses won't be repeated therefore, after printing the last response,
        # WINTER won't respond in that particular skill and directly execute the task.
        # I chose this philosophy because I hate AIs who repeat themselves a lot.
        if (response_config["last_response_idx"] + 1) <= (len(responses) - 1):
            if do_reverse:
                responses.reverse()

            if shuffle:
                # https://stackoverflow.com/a/19307329/18121288
                random.Random(shuffle_seed).shuffle(responses)

            response_config["last_response_idx"] += 1
            response = responses[response_config["last_response_idx"]]
            dprint(response) if enable_dprint else print(response)

        return response_config

    def __exec_tasks__(self, input_prompt, tasks, skillname):
        for task in tasks:
            cmd = task["cmd"]
            args = task["args"]
            exec_engine = task["execution_engine"]
            extraction_models = task["extraction_models"]
            print(skillname, cmd, args, exec_engine)
            if exec_engine == None:
                break

            for model in extraction_models:
                # evaluate the model to extract text from 'input_prompt'
                # replace that extracted text with the integer value in args.

                # evaluate model here.
                pass

            if exec_engine == "func" and skillname in self.func_dict.keys():
                self.func_dict[skillname](*args)

            elif exec_engine == "AOs":
                pass

            elif exec_engine == "skills":
                for intent in self.jsondata["skills"]:
                    if cmd == intent["skill"]:
                        tasks = intent["tasks"]
                        break

                self.__exec_tasks__(input_prompt, tasks, skillname)

            elif exec_engine == "external":
                # Look into the 'bin\powertoys' folder
                # 'cmd' will be the app/script name that 'bin\powertoys' folder,
                # 'args' will the arguments that will be passed in that app/script.
                powertoys_path = os.path.join(get_bin_path(), "powertoys")
                for file in os.listdir(powertoys_path):
                    os.system(f"{os.path.join(powertoys_path, file)} {' '.join(args)}")
