import json
from enum import Enum
import streamlit as st
from NIM import NIM

# Enum 정의
class TaskStatus(Enum):
    TODO = 1
    INPROGRESS = 2
    DONE = 3

# init
if 'nim' not in st.session_state:
    st.session_state.nim = NIM()
if 'tasks' not in st.session_state:
    st.session_state.tasks = []

# To-do board function to add new task
def add_task(new_task):
    for task in st.session_state.tasks:
        if task['name'] == new_task['name']:
            st.warning("name already exists")
            return

    new_task["status"] = TaskStatus.TODO.value
    st.session_state.tasks.append(new_task)

# To-do board function to change a task's status
def change_status(task_name, new_status):
    if new_status not in [status.value for status in TaskStatus]:
        st.warning(f"Invalid status: {new_status}. Must be one of {[status.value for status in TaskStatus]}")
        new_status = int(new_status)
    print(f"LOG change_status, new status{new_status}")
    for task in st.session_state.tasks:
        if task["name"] == task_name:
            task["status"] = new_status
            break

    st.warning(f"can't find task name {task_name}")

# To-do board function to delete a task
def delete_task(task_name):
    st.session_state.tasks = [task for task in st.session_state.tasks if task["name"] != task_name]

## To-do board function to find tasks
def find_task(search_term):
    res = []
    for task in st.session_state.tasks:
        if search_term in task['name'].lower() or search_term in task['content'].lower():
            res.append(task)
            st.success(f"LOG: name: '{task['name']}', content: '{task['content']}', status: '{task['status']}'")

    if len(res) == 0:
        st.warning(f"LOG: there is no task having {search_term}")
    return res

# Agile Board 앱 제목
st.title("To-Do board")

# =============== 작업 음성 입력 받기 ===============
def get_response(prompt):
    plan_prompt = st.session_state.nim.generate_plan_prompt(st.session_state.history, user_prompt)
    plan_res = st.session_state.nim.get_plan_response(plan_prompt)

    prev_dict = {'turn_' + str(len(st.session_state.history)): {'user_input': user_prompt, 'response': plan_res}}
    st.session_state.history.append(prev_dict)
    st.write(f"LOG history: {st.session_state.history}")
    return plan_res

def decode_plan(plan_res):
    if not plan_res or "REQUIRED" in plan_res:
        st.warning(f"LOG cannot decode plan..  {plan_res}")
        return
    try:
        plan_res = json.loads(plan_res)
    except:
        print("plan response is invalid: \n", plan_res)
        return

    # find APIs
    if plan_res["function"] == "add_task":
        #{"function": "add_task", "params": {"name": "dog hotel", "content": "REQUIRED"}}
        add_task({"name": plan_res["params"]["name"], "status": TaskStatus.TODO.value, "content": plan_res["params"]["content"]})
    elif plan_res["function"] == "find_task":
        #{"function": "find_task", "params": {"search_term": "dog"}}
        res = find_task(plan_res["params"]["search_term"])
    elif plan_res["function"] == "delete_task":
        #{"function": "delete_task", "params": {"task_name": "dog"}}
        delete_task(plan_res["params"]["task_name"])
    elif plan_res["function"] == "change_status":
        #{"function": "change_status", "params": {"task_name": "dog", "new_status": "To-Do"}}
        change_status(plan_res["params"]["task_name"], plan_res["params"]["new_status"])

if 'history' not in st.session_state:
    st.session_state.history = []

chat_container = st.container(border=True)
with chat_container:
    #history = [{'turn_0' : { 'user_input':'I want to make a note', 'response': '{"function": "add_task", "params": {"name": "REQUIRED", "content": "REQUIRED"}}'}}]
    st.write("Say something to create, delete, or find a note; ex)make a note, call it Birthday, and write it on cake pizza juice present")
    user_prompt = st.chat_input("say something: ")

    if user_prompt:
        plan_res = get_response(user_prompt)
        decode_plan(plan_res)

    if st.button("reset chat"):
        st.session_state.history = []

# 컬럼으로 상태별 작업 표시
col1, col2, col3 = st.columns([0.3,0.4,0.3])
with col1:
    st.write("### To Do")
    for task in st.session_state.tasks:
        if task["status"] == TaskStatus.TODO.value:
            container = st.container(border=True)
            with container:
                container.write(f"{task['name']}")
                container.write(f"{task['content']}")
                col11, col12 = st.columns(2)
                with col11:
                    if st.button(f"START ▶️", key=f"start_{task['name']}"):
                        change_status(task["name"], TaskStatus.INPROGRESS.value)
                        st.rerun()
                with col12:
                    if st.button(f"DELETE ❌", key=f"delete_todo_{task['name']}"):
                        delete_task(task["name"])
                        st.rerun()

with col2:
    st.write("### In Progress")
    for task in st.session_state.tasks:
        if task["status"] == TaskStatus.INPROGRESS.value:
            container = st.container(border=True)
            with container:
                container.write(f"{task['name']}")
                container.write(f"{task['content']}")
                # st.write(f"- {task['name']}")
                col21, col22, col23 = st.columns(3)
                with col21:
                    if st.button(f"UNDO ◀️️", key=f"undo_{task['name']}"):
                        change_status(task["name"], TaskStatus.TODO.value)
                        st.rerun()
                with col22:
                    if st.button(f"DONE ✅", key=f"done_{task['name']}"):
                        change_status(task["name"], TaskStatus.DONE.value)
                        st.rerun()
                with col23:
                    if st.button(f"DELETE ❌", key=f"delete_inprogress_{task['name']}"):
                        delete_task(task["name"])
                        st.rerun()

with col3:
    st.write("### Done")
    for task in st.session_state.tasks:
        if task["status"] == TaskStatus.DONE.value:
            container = st.container(border=True)
            with container:
                container.write(f"{task['name']}")
                container.write(f"{task['content']}")
                col31, col32 = st.columns(2)
                with col31:
                    if st.button(f"UNDONE ◀️", key=f"undone_{task['name']}"):
                        change_status(task["name"], TaskStatus.INPROGRESS.value)
                        st.rerun()
                with col32:
                    if st.button(f"DELETE ❌", key=f"delete_done_{task['name']}"):
                        delete_task(task["name"])
                        st.rerun()


# 모든 작업 목록 표시 (디버깅용)
st.write("### ALL TASKS LIST (DEBUGGING) below ====> ")
for task in st.session_state.tasks:
    st.write(f"- {task['name']} ({task['status']})")

# 작업 입력 받기
new_task_container = st.container(border=True)
with new_task_container:
    name = st.text_input("add your new task name")
    content = st.text_input("add your new task content")
    new_task = {"name": name, "content": content}

    if st.button("add new task"):
        if name:
            add_task(new_task)
            st.success(f"'{name}' task just added!")
            st.rerun()
        else:
            st.warning("Please, add task's name...")

    search_term = st.text_input("add any hint to find your task")
    if st.button("find task"):
        if search_term:
            decode_plan('{"function": "find_task", "params": {"search_term": "HBD"}}')
            #find_task(search_term)
            st.rerun()
        else:
            st.warning("Please, add term ...")