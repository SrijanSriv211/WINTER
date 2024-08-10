from ..shared.utils import get_bin_path
from .func import *
import random, json, os

# silicon is WINTER's execution engine.
# it will handle all of WINTER's tasks and make WINTER capable of doing automation.
class silicon:
    def __init__(self, clis_path):
        self.func_dict = {
            "set,brightness": set_brightness,
            "get,brightness": get_brightness,
            "change,brightness": change_brightness,
            "change,wallpaper": change_wallpaper,
            "search,google": search_on_google,
            "search,wikipedia": search_on_wikipedia,
            "play,video": play_video,
            "play,music": play_music,
            "search,weather": get_weather,
            "search,temperature": get_temp,
            "play,on_youtube": play_youtube_video,
            "play,random_on_youtube": play_random_youtube_video
        }

        self.clis_path = clis_path
        with open(self.clis_path, "r", encoding="utf-8") as f:
            self.clis_obj = json.load(f)

        self.__load_AOs__()

    def execute(self, command, parameters, respond=False):
        # the user just wants to chat with WINTER
        # `default` command means that there is nothing to execute.
        if command == "default":
            return

        # fetch all execution details for the command/skill
        tasks, responses, response_config = None, None, None
        for intent in self.clis_obj["clis"]:
            if command == intent["skill"]:
                tasks = intent["tasks"]
                responses = intent["responses"]
                response_config = intent["response_config"]
                break

        # respond to the user based on the details given in the response config
        response = ""
        if respond and responses != None and response_config != None:
            response, intent["response_config"] = self.__give_response__(responses, response_config)
            self.__update_response_config__()

        # execute all the tasks given in the skill sequentially
        if tasks != None:
            self.__exec_tasks__(command, parameters, tasks)

        return response if response else None
    
    def __exec_tasks__(self, command, parameters, tasks):
        for task in tasks:
            cmd = task["cmd"]
            args = task["args"]
            exec_engine = task["execution_engine"]

            if exec_engine == None:
                continue

            # map the parameters to variable arguments (denoted by a const num in `args`)
            for i, x in enumerate(args):
                if not isinstance(x, int): continue
                args[i] = parameters[x]

            self.__exec_task__(command, cmd, args, exec_engine)

    def __exec_task__(self, command, cmd, args, exec_engine):
            if exec_engine == "func" and command in self.func_dict.keys():
                self.func_dict[command](*args)

            #! Bug in AOs, where commands such as `pixelate "search anything"`
            #! basically in this the string literal " is not being removed from the string.
            #! Therefore the search query prepared by AOs is something like:
            #! https://www.google.com/search?q="search+anything"
            #! which is not good. Fix it in AOs 2.8 which is going to be built in C++.
            elif exec_engine == "AOs":
                with open(self.AOs_startup_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    lines[-1] = f"{cmd} {' '.join(args)}" + "\n."

                with open(self.AOs_startup_path, "w", encoding="utf-8") as f:
                    f.writelines(lines)

            elif exec_engine == "clis":
                for intent in self.clis_obj["clis"]:
                    if cmd == intent["skill"]:
                        tasks = intent["tasks"]
                        break

                self.__exec_tasks__(cmd, args, tasks)

    def __update_response_config__(self):
        json_obj = json.dumps(self.clis_obj, indent=4) # serializing json
        
        # writing to sample.json
        with open(self.clis_path, "w", encoding="utf-8") as f:
            f.write(json_obj)

    def __give_response__(self, responses, response_config):
        shuffle = response_config["shuffle"]
        do_reverse = response_config["reverse"]
        shuffle_seed = response_config["shuffle_seed"]

        # The response won't be selected randomly and, responses will be printed sequentially and only once.
        # Responses won't be repeated therefore, after printing the last response,
        # WINTER won't respond in that particular skill and directly execute the task.
        # I chose this philosophy because I hate AIs who repeat themselves a lot.
        if (response_config["last_response_idx"] + 1) <= (len(responses) - 1):
            if do_reverse: responses.reverse()
            if shuffle: random.Random(shuffle_seed).shuffle(responses)

            response_config["last_response_idx"] += 1
            response = responses[response_config["last_response_idx"]]

        return response, response_config

    # load AOs
    def __load_AOs__(self):
        self.AOs_path = os.path.join(get_bin_path(), "vendor\\AOs")
        self.AOs_startup_path = f"{self.AOs_path}\\Files.x72\\etc\\Startup\\exec_engine.aos"

        with open(self.AOs_startup_path, "w") as f:
            f.write(".")

        # load windows terminal
        os.system(f"wt -w 0 nt -d . cmd.exe /k \"{self.AOs_path}\\AOs.exe\"")
