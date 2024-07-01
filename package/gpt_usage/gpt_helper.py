import re,io,base64,json
from typing import List, Optional
from openai import OpenAI
from PIL import Image
from rich.console import Console
from IPython.display import Markdown,display

def printmd(string):
    display(Markdown(string)) 

class GPTchatClass:
    def __init__(
        self,
        gpt_model:str = "gpt-4",
        role_msg:str  = "Your are a helpful assistant.",
        key_path:str  = '',
        VERBOSE:bool  = True,
    ):
        self.gpt_model     = gpt_model
        self.role_msg      = role_msg
        self.key_path      = key_path
        self.VERBOSE       = VERBOSE
        self.messages      = [{"role": "system", "content": f"{role_msg}"}]
        self.init_messages = [{"role": "system", "content": f"{role_msg}"}]
        self.response      = None
        self.console       = Console()
        
        self._setup_client()

    def _setup_client(self):
        if self.VERBOSE:
            self.console.print(f"[bold cyan]key_path:[%s][/bold cyan]" % (self.key_path))

        with open(self.key_path, "r") as f:
            OPENAI_API_KEY = f.read()
        self.client = OpenAI(api_key=OPENAI_API_KEY)

        if self.VERBOSE:
            self.console.print(
                "[bold cyan]Chat agent using [%s] initialized with the follow role:[%s][/bold cyan]"
                % (self.gpt_model, self.role_msg)
            )

    def _add_message(
        self, 
        role    = "assistant", 
        content = "",
    ):
        """
        role: 'assistant' / 'user'
        """
        self.messages.append({"role": role, "content": content})

    def _get_response_content(self):
        if self.response:
            return self.response.choices[0].message.content
        else:
            return None

    def _get_response_status(self):
        if self.response:
            return self.response.choices[0].message.finish_reason
        else:
            return None

    def reset(
        self,
        role_msg:str = "Your are a helpful assistant.",
    ):
        self.init_messages = [{"role": "system", "content": f"{role_msg}"}]
        self.messages = self.init_messages

    def chat(
        self,
        user_msg         = "hi",
        PRINT_USER_MSG   = True,
        PRINT_GPT_OUTPUT = True,
        RESET_CHAT       = False,
        RETURN_RESPONSE  = False,
    ):
        self._add_message(role="user", content=user_msg)
        self.response = self.client.chat.completions.create(
            model=self.gpt_model, messages=self.messages
        )
        # Backup response for continous chatting
        self._add_message(role="assistant", content=self._get_response_content())
        if PRINT_USER_MSG:
            self.console.print("[deep_sky_blue3][USER_MSG][/deep_sky_blue3]")
            printmd(user_msg)
        if PRINT_GPT_OUTPUT:
            self.console.print("[spring_green4][GPT_OUTPUT][/spring_green4]")
            printmd(self._get_response_content())
        # Reset
        if RESET_CHAT:
            self.reset()
        # Return
        if RETURN_RESPONSE:
            return self._get_response_content()

class GPT4VchatClass:
    def __init__(
        self,
        gpt_model:str      = "gpt-4-vision-preview",
        role_msg:str       = """
            You are a helpful agent with vision capabilities; 
            do not respond to objects not depicted in images.
            """,
        image_max_size:int = 512,
        key_path:str       = '',
        VERBOSE:bool       = True,
    ):
        self.gpt_model      = gpt_model
        self.role_msg       = role_msg
        self.image_max_size = image_max_size
        self.key_path       = key_path
        self.VERBOSE        = VERBOSE
        self.messages       = [{"role": "system", "content": f"{role_msg}"}]
        self.init_messages  = [{"role": "system", "content": f"{role_msg}"}]
        self.console        = Console()
        self.response       = None
        # Setup GPT client
        self._setup_client()

    def _setup_client(self):
        """ 
            Setup GPT client
        """
        if self.VERBOSE:
            self.console.print(f"[bold cyan]key_path:[%s][/bold cyan]" % (self.key_path))

        with open(self.key_path, "r") as f:
            OPENAI_API_KEY = f.read()
        self.client = OpenAI(api_key=OPENAI_API_KEY)

        if self.VERBOSE:
            self.console.print(
                "[bold cyan]Chat agent using [%s] initialized with the follow role:[%s][/bold cyan]"
                % (self.gpt_model, self.role_msg)
            )

    def _encode_image(
        self, 
        image_pil: Image.Image
    ) -> str:
        image_pil_rgb = image_pil.convert("RGB")
        # change pil to base64 string
        img_buf = io.BytesIO()
        image_pil_rgb.save(img_buf, format="PNG")
        # Encode bytes to base64 string
        img_base64 = base64.b64encode(img_buf.getvalue()).decode("utf-8")
        return img_base64

    def _divide_by_img_tag(
        self, 
        text: str,
    ) -> List[str]:
        """
        Example:
        Input: "<img1> <img2> What is the difference of these two images?"
        Output: ['<img1>', '<img2>', ' What is the difference of these two images?']
        """

        pattern = r"(<img\d+>)"
        segments = re.split(pattern, text)
        segments = [seg for seg in segments if seg.strip() != ""]

        return segments

    def _add_message(
        self, 
        role:str = "assistant", 
        content: str = "", 
        images: Optional[List] = None,
    ):
        """
        role: 'assistant' / 'user'
        """
        if images is not None:
            # parsing text content
            image_text_segments = self._divide_by_img_tag(content)
            new_content = []
            image_num = 0
            for segment in image_text_segments:
                # check if image or text
                if segment.startswith("<img") and segment.endswith(">"):
                    # this is image
                    local_image_path = images[image_num]
                    image_pil = Image.open(local_image_path)
                    image_pil.thumbnail(
                        (self.image_max_size, self.image_max_size)
                    )
                    base64_image = self._encode_image(image_pil)
                    new_content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        }
                    )
                    image_num += 1
                else:
                    # this is text
                    new_content.append(
                        {
                            "type": "text",
                            "text": segment,
                        }
                    )
            self.messages.append({"role": role, "content": new_content})
        else:
            self.messages.append({"role": role, "content": content})

    def _get_response_content(self):
        if self.response:
            return self.response.choices[0].message.content
        else:
            return None

    def _get_response_status(self):
        if self.response:
            return self.response.choices[0].message.finish_reason
        else:
            return None

    def reset(self):
        self.messages = self.init_messages
        
    def get_state(self):
        state = self.messages
        return state
    
    def set_state(self,state):
        self.messages = state

    def chat(
        self,
        user_msg: str     = "<img> what's in this image?",
        images: List[str] = ["../img/cat.png"],
        PRINT_USER_MSG    = True,
        PRINT_GPT_OUTPUT  = True,
        RESET_CHAT        = False,
        RETURN_RESPONSE   = True,
        MAX_TOKENS        = 512,
    ):
        self._add_message(role="user", content=user_msg, images=images)
        self.response = self.client.chat.completions.create(
            model=self.gpt_model, messages=self.messages, max_tokens=MAX_TOKENS
        )
        # Backup response for continous chatting
        self._add_message(role="assistant", content=self._get_response_content())
        if PRINT_USER_MSG:
            self.console.print("[deep_sky_blue3][USER_MSG][/deep_sky_blue3]")
            printmd(user_msg)
        if PRINT_GPT_OUTPUT:
            self.console.print("[spring_green4][GPT_OUTPUT][/spring_green4]")
            printmd(self._get_response_content())
        # Reset
        if RESET_CHAT:
            self.reset()
        # Return
        if RETURN_RESPONSE:
            return self._get_response_content()

# For MavenClass usages
class DataLogClass(object):
    def __init__(
            self,
            t, # ndarray [L]
            x, # ndarray [L x d]
            system_prompt = None,
        ):
        self.t = t
        self.x = x
        self.d = self.x.shape[1]
        self.dt = self.t[1]-self.t[0]
        # Concatenate previous log histories 
        self.messages = []
        # Append the initial system prompt
        if system_prompt is not None:
            self.append_messages(role='system',content=system_prompt)

    def append_messages(
        self,
        role         = 'system',
        content      = None,
        name         = None,
        tool_call_id = None,
    ):
        """ 
            Append user message
        """
        self.messages.append({'role':role,'content':content})
        if name is not None: self.messages[-1]['name'] = name
        if tool_call_id is not None: self.messages[-1]['tool_call_id'] = tool_call_id

class MavenClass(object):
    def __init__(
        self,
        client,
        datalog,
        tools,
        function_mappings,
        model='gpt-4-1106-preview'
    ):
        """ 
        Initialize Maven, an intelligent API recommender based on GPTs
        :param client: OpenAI client
        :param datalog: This contains data as well as log for OpenAI APIs
        """
        self.client            = client
        self.datalog           = datalog
        self.tools             = tools
        self.function_mappings = function_mappings
        self.model             = model

    def recommend_api(
        self,
        usr_msg,
        n_choice = 1,
        verbose  = True,
    ):
        """ 
        Recomment APIs from user meassage
        for tool create: https://platform.openai.com/docs/api-reference/chat/create#chat-create-tools 
        for json schema: https://json-schema.org/understanding-json-schema/reference/type
        """
        # Append user message to datalog
        self.datalog.append_messages(role='user',content=usr_msg)

        # Get response from the server
        response = self.client.chat.completions.create(
            model       = self.model,
            messages    = self.datalog.messages,
            tools       = self.tools,
            tool_choice = 'auto',
            n           = n_choice,
        )
        self.recommended_api = response
        if verbose:
            self.print_recommended_api()

    def print_recommended_api(self):
        """ 
            Print recommended APIs
        """
        recommended_api = self.recommended_api
        n_choice = len(recommended_api.choices)
        for c_idx in range(n_choice): # for each choice
            choice     = recommended_api.choices[c_idx]
            tool_calls = choice.message.tool_calls
            if tool_calls:
                for tool_idx,tool_call in enumerate(tool_calls): # for each tool
                    function_name = tool_call.function.name
                    function_args = tool_call.function.arguments
                    print ("choice:[%d/%d] tool_idx:[%d/%d] function_name:[%s] function_args:%s"%
                        (c_idx,n_choice,tool_idx,len(tool_calls),function_name,function_args))
                if (n_choice>1) and (c_idx<(n_choice-1)):
                    print ("")
    
    def select_recommended_api(
        self,
        choice_idx = 0,
        verbose    = True,
    ):
        """ 
            Select one of the recommended api
        """
        print ("choice_idx:[%d] selected"%(choice_idx))
        recommended_api = self.recommended_api
        choice          = recommended_api.choices[choice_idx]
        tool_calls      = choice.message.tool_calls
        for tool_idx,tool_call in enumerate(tool_calls): # for each tool
            function_name = tool_call.function.name
            function_args = tool_call.function.arguments
            if verbose: # print first
                print ("tool_idx:[%d/%d] function_name:[%s] function_args:%s"%
                    (tool_idx,len(tool_calls),function_name,function_args))
            # Function call
            _fn = self.function_mappings[function_name]
            json_function_args = json.loads(function_args)
            _fn_output = _fn(data=self.datalog,**json_function_args) # actual function call happens here
            # Append message
            self.datalog.append_messages(
                role    = 'assistant',
                content = str(tool_call.function),
            )
            self.datalog.append_messages(
                role         = 'function',
                content      = _fn_output,
                name         = function_name,
                tool_call_id = tool_call.id,
            )

