import os

from args_agent import Args as args
import anthropic
import sys
from ddgs import DDGS
import inspect

TOOLS = [{"name": "search",
          "description": ""}]

# 基于anthropic的agent_loop
class AgentAnthropic:
    def __init__(self, prompt):
        self.prompt = prompt
        self.model = os.environ.get("DEEPSEEK_MODEL", args.model)
        self.message_history = []
        self.api_key = os.environ.get("API_KEY", args.API_KEY)
        self.base_url = os.environ.get("ANTHROPIC_BASE_URL", args.DEFAULT_BASE_URL)
        self.client = None

    def reset_history(self):
        self.message_history = []

    def pull_model(self):
        api_key = self.api_key
        if api_key and not api_key.startswith("sk-"):
            raise SystemExit(" API_KEY 密钥异常，密钥应以 sk- 开头 ")
        if api_key:
            if self.base_url == args.DEFAULT_BASE_URL:
                print(" = = = 云端模型拉取失败，改用Ollama 本地模型 = = = ")
            client = anthropic.Anthropic(api_key = api_key, base_url = self.base_url)
            self.client = client
            print(f"{self.model}模型初始化成功\n (˘ω˘)Zzz  -->  ヽ (^∀^)ﾉ ")
            # return client
        else:
            sys.exit(" 密钥获取失败，请检查相关文件 ")

    def generate_response(self, user_ask: str, tools_show: list = None):
        self.message_history.append({"role": "user", "content": user_ask})
        try:
            # print("本次发送完整messages")
            # print(json.dumps(self.message_history, ensure_ascii=False, indent=2))
            response = self.client.messages.create(model = self.model,
                                         max_tokens = 5000,  # 设置太少思考会用完
                                         system = self.prompt,
                                         messages = self.message_history,
                                         tools = tools_show)
            # 遍历解析，把thinking内容 + 正式回答转为字符串
            full_text = ""
            for block in response.content:
                if hasattr(block, "thinking") and block.thinking is not None:
                    full_text += block.thinking
                if hasattr(block, "text") and block.text is not None:
                    full_text += block.text
            self.message_history.append({"role": "assistant", "content": full_text})
            # self.message_history.append({"role": "assistant", "message": response.content})
            return response
        except anthropic.APIConnectionError as error:
            err_link = f"连接模型接口失败: {error}"
            print(err_link)
            return f"错误：{err_link}"
        except anthropic.APIStatusError as error:
            err_status = f"接口返回错误: {error.status_code} \n请检查 Key、余额和模型名"
            print(err_status)
            return f"错误：{err_status}"
        except Exception as error:
            err_other = f"调用 LLM 出现未知错误: {error}"
            print(err_other)
            return f"错误：{err_other}"

    def run_agent_loop(self, user_ask: str, max_steps: int, tools_show: list, tools: dict):
        self.pull_model()
        for step in range(1, max_steps+1):
            process = step / max_steps
            print(f" = = = = = = 第{step}/{max_steps}轮思考 --> 思考深度[{process:.0%}] = = = = = = \n")
            response = self.generate_response(user_ask, tools_show)
            for res in response.content:
                if res.type == 'text':
                    print(res.text)
            # using_tools = [block for block in response.content if block.type == 'tool_use']
            using_tools = any(block.type == 'tool_use' for block in response.content)  # Ture of False
            if not using_tools:
                return
            results = []
            for res in response.content:
                if res.type == 'tool_use':
                    print(f"\033[33m> 调用工具: {res.name} \033[0m")
                    handler = tools.get(res.name, '')
                    print("工具入参 raw input：", res.input)  # 新增调试
                    output = handler(**res.input) if handler else f"没有{res.name}这个工具\n请检查是否添加{res.name}工具"
                    print(f"\033[36mAI >> {str(output or '')[:300]}\033[0m")
                    results.append({"type": "tool_result", "tool_use_id": res.id, "content": output})
            self.message_history.append({"role": "user", "content": results})


# 工具类
class ToolExecutor:
    def __init__(self):
        self.Tools = {}
        self.Tools_show = []
    def register_tool(self, name: str , description: str, func):
        sig = inspect.signature(func)
        if not isinstance(sig, inspect.Signature):
            raise TypeError("sig 必须是 inspect.Signature 类型")
        json_type = "string"
        properties = {}
        required_list = []
        for param_name, p_info in sig.parameters.items():
            if p_info.annotation is not inspect._empty:
                if hasattr(p_info.annotation, "__name__"):
                    type_str = p_info.annotation.__name__
                    # 适配标准JSON Schema关键字 --> LLM
                    if type_str == "str":
                        json_type = "string"
                    elif type_str == "int":
                        json_type = "integer"
                    elif type_str == "bool":
                        json_type = "boolean"
                    else:
                        json_type = "string"

            properties[param_name] = {"type": json_type}
            if p_info.default == inspect.Parameter.empty:
                required_list.append(param_name)

        tools_index = next((idx for idx, t in enumerate(self.Tools_show) if name == t["name"]), None)
        if tools_index is not None:
            a = input(f"警报！ 工具 {name} 已存在，是否覆盖( y / n ): ").strip().lower()[0:1]
            while a not in ['y', 'n']:
                print(f"错误！ 不存在{a}选项，请重新输入 \n (；¬＿¬) -- (▔︵▔)")
                a = input(f"警报！ 工具 {name} 已存在，是否覆盖( y / n ): ").strip().lower()[0:1]
            if a == "y":
                self.Tools_show.append({"name": name, "description": description,
                                       "input_schema": {"type": "object",
                                                        "properties": properties,
                                                        "required": required_list}})
                self.Tools[name] = func
            else:
                print(f"取消覆盖原{name}工具")
                return
        else:
            self.Tools_show.append({"name": name, "description": description,
                                    "input_schema": {"type": "object",
                                                     "properties": properties,
                                                     "required": required_list}})
            self.Tools[name] = func
        text = f"成功添加 {name} 工具 \n (✧ω✧) -- (๑˃̵ᴗ˂̵)و -- ╰(✧∇✧)╯"
        print(f"\033[32m{text}\033[0m")

    def display_tools(self):
        show = []
        for idx, s in enumerate(self.Tools_show):
            show.append(f"[{idx + 1}] {s.get('name', '')}\n    {s.get('description', '')}")
        exhibition = '\n'.join(show)
        show_result = "= = = = = = = = = < 当前可调用工具 > = = = = = = = = = \n" + exhibition + "\n= = = = = = = = = = = = = = = = = = = = = = = = = = "
        print(show_result)


# example to test
def search(query: str) -> str:
    # 无密钥DuckDuckGo网页搜索 --> 禁止频繁调用    free of charge
    print(f" (๑・̀ㅂ・́)و✧查询中 --> 正在执行 [DuckDuckGo] 网页搜索 < {query} > ")
    try:
        with DDGS(timeout=20) as ddgs:
            results = list(ddgs.text(
                query = query,
                region = "cn-zh",
                safesearch = "moderate",  # strict <严格过滤成人内容>  moderate <中等>  off <关闭>
                max_results = 3
            ))
        if not results:
            return f"对不起，没有找到关于 '{query}' 的信息 \n (´･_･`) -- (；´Д｀) -- (´°̥̥̥ω°̥̥̥`) -- (＞＜;) "
        snippets = []
        for idx, res in enumerate(results):
            title = res.get("title", "")
            content = res.get("body", "")
            snippets.append(f"[{idx + 1}] {title}\n{content}")
        return "\n\n".join(snippets)
    except Exception as e:
        err =  f"搜索时发生错误: {e}"
        print(err)
        return f"错误：{err}"


if __name__ == "__main__":
    executor = ToolExecutor()
    t_name = 'search'
    t_description = "执行网页搜索，获取互联网公开信息。适用于需要实时资料、外部知识、事实查询的场景。参数query为搜索关键词。最多返回3条结果。"
    executor.register_tool(name = t_name, description = t_description, func = search)
    executor.display_tools()
    tools_show = executor.Tools_show
    tools = executor.Tools

    prompt = None
    if prompt is None:
        prompt = args.prompt
    max_step = None
    if max_step is None:
        max_step = args.max_step
    agent = AgentAnthropic(prompt)

    print("============== 输入 exit | q 退出对话 ==============")
    while True:
        try:
            user_ask = input("\033[36m你（用户） >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if user_ask.strip().lower() in ('exit', 'q'):
            print("好的，感谢使用 (*´∀`)~♥")
            break
        agent.run_agent_loop(user_ask, max_step, tools_show, tools)


