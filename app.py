import streamlit as st
import os
import json
import uuid
from google import genai
from google.genai import types
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

# --- Custom MCP Primitives (No changes needed here) ---
@dataclass
class ToolCall:
    """Simulates the Model Context Protocol ToolCall structure."""
    name: str
    invocationId: str = field(default_factory=lambda: str(uuid.uuid4()))
    args: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ToolResult:
    """Simulates the Model Context Protocol ToolResult structure."""
    toolName: str
    invocationId: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# --- 1. Configuration and Setup ---
# Set API Key from environment or Streamlit secrets
API_KEY = os.environ.get("GEMINI_API_KEY", st.secrets.get("GEMINI_API_KEY", ""))
MODEL_NAME = "gemini-2.5-flash-preview-09-2025" # Use a stable tool-use model

st.set_page_config(layout="wide", page_title="Gemini Agent with Simulated MCP Tool")

if API_KEY:
    try:
        client = genai.Client(api_key=API_KEY)
    except Exception as e:
        st.error(f"Failed to initialize Gemini Client: {e}")
        client = None
else:
    client = None

# --- 2. The Simulated MCP Server (External Data Source) ---
class MyCustomMCPServer:
    # This is the data the agent is designed to access
    PRIVATE_DATA = {
        "project_status": "The 'Quantum Leap' project is 85% complete. Final testing scheduled for 2026-03-15.",
        "internal_contact": "Lead Engineer: Dr. Anya Sharma (ext: 5891)",
        "security_policy": "All unreleased technical specifications are classified Level-3 and cannot be shared externally."
    }

    # The formal tool specification for the LLM to read
    MCP_TOOL_SPEC = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="get_internal_status",
                description="Retrieves the current status and key details from the company's internal project management system. Use this to answer questions about internal data, projects, or contacts.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "data_key": types.Schema(
                            type=types.Type.STRING,
                            description="The specific piece of data to retrieve (e.g., 'project_status', 'internal_contact', 'security_policy')."
                        )
                    },
                    required=["data_key"],
                ),
            )
        ]
    )

    def execute_tool(self, call: ToolCall) -> ToolResult:
        """Executes the tool call and returns an MCP ToolResult."""
        if call.name == "get_internal_status":
            try:
                requested_key = call.args.get("data_key", "").lower().replace(' ', '_')
                
                # Robustly map the LLM's requested key to our internal data keys
                if "project" in requested_key or "status" in requested_key:
                    data_key = "project_status"
                elif "contact" in requested_key or "engineer" in requested_key:
                    data_key = "internal_contact"
                elif "policy" in requested_key:
                    data_key = "security_policy"
                else:
                    data_key = None
                
                if data_key and data_key in self.PRIVATE_DATA:
                    result_data = self.PRIVATE_DATA[data_key]
                    return ToolResult(
                        toolName=call.name,
                        invocationId=call.invocationId,
                        result={"status_data": result_data}
                    )
                else:
                    return ToolResult(
                        toolName=call.name,
                        invocationId=call.invocationId,
                        error=f"Error: Data key '{call.args.get('data_key')}' not found. Available keys: {list(self.PRIVATE_DATA.keys())}"
                    )
            except Exception as e:
                return ToolResult(
                    toolName=call.name,
                    invocationId=call.invocationId,
                    error=f"Execution error: {str(e)}"
                )
        else:
            return ToolResult(
                toolName=call.name,
                invocationId=call.invocationId,
                error=f"Tool not implemented: {call.name}"
            )

# --- 3. Agent Workflow Function ---
def run_mcp_agent_workflow(prompt: str, server: MyCustomMCPServer):
    if not client:
        st.error("Client not initialized. Check API key.")
        return

    st.subheader("🤖 Agent 1: The MCP Agent")
    
    # 1. FIRST TURN: Send the prompt and enable tools
    st.info("Agent is processing prompt and checking for tool need...")
    first_response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[prompt],
        config=types.GenerateContentConfig( 
            system_instruction="You are a specialized Internal Information Agent. You must use the 'get_internal_status' tool to answer any questions related to company projects, status, or contacts. If a question is external (e.g., 'What is the weather?'), answer directly without using the tool.",
            tools=[server.MCP_TOOL_SPEC]
        )
    )
    
    # Check if a tool call was made
    if not first_response.function_calls:
        st.subheader("✅ Final Agent Response (No Tool Used)")
        st.markdown(first_response.text)
        return

    # --- Tool Call Loop Initiated ---
    st.info("Agent detected a need for external data. Executing tool...")

    # Build the list of tool results
    tool_parts_list: List[types.Part] = []
    
    for call in first_response.function_calls:
        st.markdown("**➡️ Tool Call Detected:**")
        st.code(f"Tool: {call.name} | Args: {dict(call.args)}", language="json")
        
        # 2. Execute the simulated MCP server tool
        invocation_id = str(uuid.uuid4())
        mcp_tool_call = ToolCall(
            name=call.name,
            invocationId=invocation_id,
            args=dict(call.args)
        )
        mcp_result: ToolResult = server.execute_tool(mcp_tool_call)
        
        if mcp_result.result:
            result_content = mcp_result.result
        else:
            result_content = {"error_message": mcp_result.error}
        
        # Create the Part object for the tool result
        gemini_tool_part = types.Part.from_function_response( 
            name=mcp_result.toolName,
            response=result_content
        )
        tool_parts_list.append(gemini_tool_part)

        st.markdown(f"**⬅️ Tool Result:**")
        st.code(json.dumps(result_content, indent=2), language='json')
        
    # 3. SECOND TURN: Send the *entire conversation* plus tool results for the final answer
    st.info("Sending full history and tool results back to the Agent for final answer generation...")

    # 🛑 CRITICAL FIX: Correcting the list structure and commas for the contents_for_second_turn list.
    contents_for_second_turn = [
        types.Content(role="user", parts=[types.Part.from_text(prompt)]), # Line 1: User content
        first_response.candidates[0].content,                            # Line 2: Model's function call (Content object)
        types.Content(role="tool", parts=tool_parts_list)                # Line 3: Tool's result
    ]
    
    final_response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents_for_second_turn,
        config=types.GenerateContentConfig( 
            system_instruction="You are a specialized Internal Information Agent. You must use the 'get_internal_status' tool to answer any questions related to company projects, status, or contacts. If a question is external (e.g., 'What is the weather?'), answer directly without using the tool.",
            tools=[server.MCP_TOOL_SPEC]
        )
    )

    # 4. Final response
    st.subheader("✅ Final Agent Response")
    st.markdown(final_response.text)


# --- 4. Streamlit UI ---

st.title("Gemini Agent with Model Context Protocol (MCP) Simulation")

st.markdown("""
<div style="padding: 10px; background-color: #f0fff0; border-radius: 8px; border: 1px solid green;">
    ✅ **Syntax Fix:** The final `TypeError` was due to a **missing comma and improper list structure** on the line defining the `contents_for_second_turn`. The list syntax has been corrected, which is the last known point of failure in the complex, multi-turn tool-use logic.
</div>
""", unsafe_allow_html=True)

st.divider()

if not API_KEY:
    st.warning("Please set your Gemini API Key in the Streamlit Cloud Secrets (`GEMINI_API_KEY`) or as an environment variable.")

# Initialize the simulated MCP server
mcp_server_instance = MyCustomMCPServer()

prompt = st.text_input(
    "Ask the Internal Agent a question:",
    placeholder="e.g., 'What is the status of the Quantum Leap project?' (This requires the tool)",
    key="prompt_input"
)

# Display the data the agent has access to
with st.expander("🔐 Internal Data (MCP Server Data)"):
    st.markdown("The Agent **only** knows this data if it successfully calls the `get_internal_status` tool.")
    st.json(mcp_server_instance.PRIVATE_DATA)

if st.button("Run MCP Agent", type="primary") and prompt:
    if client:
        with st.spinner("Executing Agent Workflow..."):
            run_mcp_agent_workflow(prompt, mcp_server_instance)
    else:
        st.error("Cannot run. Gemini Client is not initialized.")
