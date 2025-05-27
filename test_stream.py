import os
import streamlit as st
from dashscope import Assistants, Messages, Runs, Threads
import json
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

st.set_page_config(page_title="OLED实验室制备助手", page_icon="🧪", layout="wide")
st.title("🧪 有机发光二极管(OLED)实验室制备助手")

# 分子指纹相关函数
def calculate_morgan_fingerprint(smiles, radius=2, nBits=2048):
    """计算分子的Morgan指纹"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # 使用新的 MorganGenerator 替代弃用的 GetMorganFingerprintAsBitVect
            morgan_gen = GetMorganGenerator(radius=radius, fpSize=nBits)
            fp = morgan_gen.GetFingerprint(mol)
            return np.array(fp)
        else:
            return np.zeros(nBits)
    except:
        return np.zeros(nBits)

def calculate_maccs_fingerprint(smiles):
    """计算分子的MACCS指纹"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = MACCSkeys.GenMACCSKeys(mol)
            return np.array(fp)
        else:
            return np.zeros(167)  # MACCS指纹长度为167
    except:
        return np.zeros(167)

def search_similar_molecules(query_smiles, fp_type='morgan', top_n=5):
    
    """根据分子指纹相似性搜索数据库中的相似分子"""
    # 计算查询分子的指纹
    if fp_type == 'morgan':
        query_fp = calculate_morgan_fingerprint(query_smiles)
        fp_col = 'morgan_fp'
    else:  # maccs
        query_fp = calculate_maccs_fingerprint(query_smiles)
        fp_col = 'maccs_fp'
    
    # 读取数据
    db_path ='./data.pkl'

    db_data = pd.read_pickle(db_path)
        

    # 计算相似性（使用Tanimoto系数）
    def calculate_tanimoto(fp):
        # fp是numpy数组
        # 计算Tanimoto系数
        intersection = np.sum(np.logical_and(query_fp, fp))
        union = np.sum(np.logical_or(query_fp, fp))
        if union == 0:
            return 0.0
        return intersection / union
    
    db_data['similarity'] = db_data[fp_col].apply(calculate_tanimoto)
    
    # 按相似性降序与Maximum EQE降序排序并返回top_n个结果
    results = db_data.sort_values('similarity', ascending=False).head(top_n)
    # 再按照maximum_EQE降序排序
    # maximum_EQE设置为float类型
    results['maximum_EQE_value'] = results['maximum_EQE_value'].astype(float)
    results = results.sort_values('maximum_EQE_value', ascending=False).head(1)
    # 只选择第一个
    return results[['material_SMILES', 'DOI', 'similarity', 'anode', 'hole_injection_layer', 'hole_transport_layer',
       'emission_layer_details', 'emission_layer_type', 'host', 'dopants',
       'emission_layers', 'electron_transport_layer',
       'electron_injection_layer', 'cathode', 'device_emission_wavelength',
       'device_brightness', 'turn_on_voltage', 'current_efficiency',
       'power_efficiency', 'maximum_EQE', 'device_lifetime',
       'device_emission_wavelength_value', 'device_emission_wavelength_unit',
       'device_brightness_value', 'device_brightness_unit',
       'turn_on_voltage_value', 'turn_on_voltage_unit',
       'current_efficiency_value', 'current_efficiency_unit',
       'power_efficiency_value', 'power_efficiency_unit', 'maximum_EQE_value',
       'maximum_EQE_unit', 'device_lifetime_value', 'device_lifetime_unit',
       'pure_emitter', 'dopants_wt_percent', 'dopants_name', 'material_name']]
    # return {"material_SMILES": ['CC(C)(C)c1ccc2c(c1)c1cc(C(C)(C)C)cc3c1n2-c1ccc2c4c1B3c1cc(C(C)(C)C)cc3c5cc(C(C)(C)C)cc(c5n-4c13)S2'],'similarity': [0.9999999999999999]}

# 侧边栏 - API Key 设置
with st.sidebar:
    st.header("设置")
    
    # 从环境变量或会话状态获取 API Key
    default_api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if "api_key" not in st.session_state:
        st.session_state.api_key = default_api_key
    
    # API Key 输入
    api_key = st.text_input(
        "DashScope API Key", 
        value=st.session_state.api_key,
        type="password",
        help="输入您的 DashScope API Key。您可以在 DashScope 控制台获取: https://dashscope.console.aliyun.com/"
    )
    
    # 保存 API Key 到会话状态
    if api_key != st.session_state.api_key:
        st.session_state.api_key = api_key
        # 重置会话状态
        if "messages" in st.session_state:
            st.session_state.messages = []
            st.rerun()
    
    # 知识库ID输入
    if "knowledge_base_id" not in st.session_state:
        st.session_state.knowledge_base_id = "lbuni0sw84"
    
    knowledge_base_id = st.text_input(
        "知识库ID", 
        value=st.session_state.knowledge_base_id,
        help="输入您的知识库ID"
    )
    
    if knowledge_base_id != st.session_state.knowledge_base_id:
        st.session_state.knowledge_base_id = knowledge_base_id
        # 重置会话状态
        if "assistant_id" in st.session_state:
            del st.session_state.assistant_id
            del st.session_state.thread
            if "messages" in st.session_state:
                st.session_state.messages = []
            st.rerun()
    
    # 模型选择
    model = st.selectbox(
        "选择模型",
        ["qwen-plus", "qwen-max", "qwen-turbo"],
        index=0,
        help="选择要使用的 DashScope 模型"
    )
    
    # 分子指纹类型选择
    fp_type = st.selectbox(
        "分子指纹类型",
        ["morgan", "maccs"],
        index=0,
        help="选择用于分子相似性搜索的指纹类型"
    )
    
    # 相似分子返回数量
    top_n = st.slider(
        "返回相似分子数量", 
        min_value=1, 
        max_value=20, 
        value=5,
        help="设置相似性搜索返回的分子数量"
    )
    
    # 显示使用说明
    st.markdown("### 使用说明")
    st.markdown("""
    1. 输入您的 DashScope API Key
    2. 输入您的知识库ID
    3. 在聊天框中输入关于OLED制备的问题
    4. 输入SMILES字符串并使用"搜索相似分子"功能
    """)

# 验证 API Key
if not st.session_state.api_key:
    st.warning("请在侧边栏输入您的 DashScope API Key 以继续使用助手。")
    st.stop()

# 设置 API Key 到环境变量
os.environ["DASHSCOPE_API_KEY"] = st.session_state.api_key

def create_assistant(index_id, model_name="qwen-plus"):
    """创建一个使用指定知识库的 Assistant。"""
    assistant = Assistants.create(
        model=model_name,  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        name='有机发光二极管(OLED)实验室制备助手',
        description='一个有机发光二极管(OLED)实验室制备助手',
        instructions='你是一个有机发光二极管(OLED)实验室制备助手，擅长器件制备，可以回答一切关于OLED制备的问题。使用提供的知识库来回答用户的问题。以下信息可能对你有帮助：${documents}。当用户提供SMILES字符串时，你可以使用分子相似性搜索功能查找相似的分子。',
        tools=[
            {
                "type": "rag",  # 指定使用RAG（检索增强生成）模式
                "prompt_ra": {
                    "pipeline_id": [index_id],  # 指定使用的知识库索引ID
                    "multiknowledge_rerank_top_n": 10,  # 多知识源重排序时返回的top N结果数
                    "rerank_top_n": 5,  # 最终重排序后返回的top N结果数
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query_word": {
                                "type": "str",
                                "value": "${documents}"  # 使用动态占位符，将被实际查询内容替换
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_similar_molecules",
                    "description": "根据SMILES字符串搜索相似的分子",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "smiles": {
                                "type": "string",
                                "description": "分子的SMILES字符串"
                            },
                            "fp_type": {
                                "type": "string",
                                "enum": ["morgan", "maccs"],
                                "description": "使用的分子指纹类型"
                            },
                            "top_n": {
                                "type": "integer",
                                "description": "返回的相似分子数量"
                            }
                        },
                        "required": ["smiles"]
                    }
                }
            }
        ]
    )
    return assistant.id

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

# 初始化 Assistant 和 Thread
if "assistant_id" not in st.session_state:
    with st.spinner("正在初始化助手..."):
        try:
            assistant_id = create_assistant(st.session_state.knowledge_base_id, model)
            st.session_state.assistant_id = assistant_id
            st.success(f"助手初始化成功！ID: {assistant_id}")
        except Exception as e:
            st.error(f"初始化助手失败: {str(e)}")
            st.stop()

if "thread" not in st.session_state:
    with st.spinner("正在创建对话线程..."):
        try:
            thread = Threads.create()
            st.session_state.thread = thread
            st.success("对话线程创建成功！")
        except Exception as e:
            st.error(f"创建对话线程失败: {str(e)}")
            st.stop()

# 显示聊天历史
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            st.write(message["content"])
            # 如果有工具调用，显示工具调用信息
            if "tool_calls" in message and message["tool_calls"]:
                for tool_call in message["tool_calls"]:
                    with st.status(f"工具调用: {tool_call.get('name', '知识库检索')}", state="complete"):
                        st.write("调用参数:")
                        if "args" in tool_call:
                            st.json(tool_call["args"])
                        else:
                            st.write(tool_call.get("content", "无调用参数"))
                        
                        if "output" in tool_call:
                            st.write("调用结果:")
                            st.write(tool_call["output"])


# 获取用户输入
if prompt := st.chat_input("请输入您的问题..."):
    # 添加用户消息到历史记录
    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        st.write(prompt)
    
    # 显示AI思考中状态
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("思考中...")
        
        # 创建一个容器用于显示检索状态
        retrieval_status_container = st.container()
        
        try:
            # 创建用户消息
            Messages.create(thread_id=st.session_state.thread.id, content=prompt)
            
            # 使用原生流式输出创建运行
            run = Runs.create(
                thread_id=st.session_state.thread.id, 
                assistant_id=st.session_state.assistant_id,
                stream=True  # 启用流式输出
            )

            # 初始化变量
            full_response = ""
            tool_calls_info = []
            is_tool_call = False
            tool_call_statuses = {}  # 存储工具调用状态组件

            # 使用 while 循环来处理可能的多轮工具调用
            while True:
                # 处理流式输出
                for event, data in run:
                    # 处理消息增量更新
                    if event == 'thread.message.delta':
                        if hasattr(data, 'delta') and hasattr(data.delta, 'content'):
                            if hasattr(data.delta.content, 'text') and hasattr(data.delta.content.text, 'value'):
                                # 更新消息内容
                                text_delta = data.delta.content.text.value
                                full_response += text_delta
                                message_placeholder.markdown(full_response + "▌")
                    
                    # 处理工具调用
                    elif event == 'thread.run.requires_action':
                        is_tool_call = True
                        
                        # 收集工具调用信息
                        tool_outputs = []
                        for tool_call in data.required_action.submit_tool_outputs.tool_calls:
                            tool_call_id = tool_call.id
                            
                            # 创建工具调用状态
                            with retrieval_status_container:
                                tool_call_name = tool_call.function.name if hasattr(tool_call, 'function') else "知识库检索"
                                tool_call_statuses[tool_call_id] = st.status(
                                    f"工具调用: {tool_call_name}", 
                                    state="running"
                                )
                                
                                # 显示调用参数
                                with tool_call_statuses[tool_call_id]:
                                    st.write("调用参数:")
                                    try:
                                        args_json = json.loads(tool_call.function.arguments)
                                        st.json(args_json)
                                    except:
                                        st.write(tool_call.function.arguments)
                            
                            # 收集工具调用信息
                            tool_call_info = {
                                "id": tool_call_id,
                                "type": tool_call.type,
                                "name": tool_call.function.name if hasattr(tool_call, 'function') else "知识库检索",
                                "args": tool_call.function.arguments if hasattr(tool_call, 'function') else "",
                                "content": tool_call.function.arguments if hasattr(tool_call, 'function') else ""
                            }
                            tool_calls_info.append(tool_call_info)
                            
                            # 处理不同类型的工具调用
                            if hasattr(tool_call, 'function') and tool_call.function.name == "search_similar_molecules":
                                # 处理分子相似性搜索
                                args = json.loads(tool_call.function.arguments)
                                smiles = args.get("smiles", "")
                                fp_type_arg = args.get("fp_type", fp_type)  # 使用参数或默认值
                                top_n_arg = args.get("top_n", top_n)  # 使用参数或默认值
                                
                                results = search_similar_molecules(
                                    query_smiles=smiles,
                                    fp_type=fp_type_arg,
                                    top_n=top_n_arg
                                )
                                
                                # 将结果转换为适当的格式
                                if isinstance(results, dict):
                                    results_df = pd.DataFrame(results)
                                    output = {"results": results_df.to_dict(orient='records')}
                                else:
                                    # 如果已经是DataFrame，直接使用
                                    output = {"results": results.to_dict(orient='records')}
                                
                                tool_call_info["output"] = json.dumps(output)
                                
                                tool_outputs.append({
                                    "tool_call_id": tool_call_id,
                                    "output": json.dumps(output)
                                })
                            else:
                                # 默认为知识库检索
                                tool_outputs.append({
                                    "tool_call_id": tool_call_id,
                                    "output": "检索成功"
                                })
                        
                        # 更新工具调用状态
                        for tool_call in tool_calls_info:
                            if tool_call["id"] in tool_call_statuses:
                                with tool_call_statuses[tool_call["id"]]:
                                    st.write("调用结果:")
                                    if "output" in tool_call:
                                        try:
                                            output_json = json.loads(tool_call["output"])
                                            if "results" in output_json:
                                                # 显示相似分子结果
                                                results_df = pd.DataFrame(output_json["results"])
                                                st.dataframe(results_df)
                                            elif "error" in output_json:
                                                st.error(output_json["error"])
                                            else:
                                                st.write(tool_call["output"])
                                        except:
                                            st.write(tool_call["output"])
                                    else:
                                        st.write("调用完成")
                                tool_call_statuses[tool_call["id"]].update(state="complete")
                        
                        # 提交工具输出，并获取新的 run 对象
                        try:
                            run = Runs.submit_tool_outputs(
                                thread_id=st.session_state.thread.id,
                                run_id=data.id,
                                tool_outputs=tool_outputs,
                                stream=True  # 启用流式输出
                            )
                            # 跳出当前 for 循环，使用新的 run 对象继续处理事件流
                            break
                        except Exception as e:
                            st.error(f"提交工具输出失败: {str(e)}")
                            import traceback
                            st.error(traceback.format_exc())
                            break
                    
                    # 处理运行完成事件
                    elif event == 'thread.run.completed':
                        # 如果没有完整响应，获取最终消息
                        if not full_response:
                            # 获取消息列表
                            msgs = Messages.list(st.session_state.thread.id)
                            if msgs and 'data' in msgs and len(msgs['data']) > 0:
                                final_reply = msgs['data'][0]['content'][0]['text']['value']
                                message_placeholder.markdown(final_reply)
                                full_response = final_reply
                        # 运行完成，退出循环
                        break
                    
                    # 处理运行失败事件
                    elif event in ['thread.run.failed', 'thread.run.cancelled', 'thread.run.expired']:
                        message_placeholder.error(f"运行失败: {event}")
                        break
                else:
                    # for 循环正常结束（没有触发 break），说明所有事件都已处理完毕
                    break
                
                # 检查是否需要继续处理新的 run 对象
                if event != 'thread.run.requires_action':
                    # 如果最后一个事件不是工具调用，说明不需要继续处理
                    break
            
            # 移除光标
            message_placeholder.markdown(full_response)
            
            # 添加助手回复到历史记录
            assistant_message = {
                "role": "assistant", 
                "content": full_response,
                "tool_calls": tool_calls_info
            }
            st.session_state.messages.append(assistant_message)
            
        except Exception as e:
            st.error(f"处理请求时出错: {str(e)}")
            import traceback
            st.error(traceback.format_exc())  # 显示详细错误信息，帮助调试
            st.warning("请检查您的 API Key 和知识库ID是否正确，或者 DashScope 服务是否可用。")
