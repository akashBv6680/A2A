import streamlit as st
import os
import json
import uuid
from google import genai
from google.genai import types
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

# --- Custom MCP Primitives (Replaces 'modelcontextprotocol' dependency) ---
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
MODEL_NAME = "gemini-2.5-flash-preview-09-2025"

st.set_page_config(layout="wide", page_title="Gemini Agent with Simulated MCP Tool")

if API_KEY:
    try:
        # Initialize client with the API key
        client = genai.Client(api_key=API_KEY)
    except Exception as e:
        st.error(f"Failed to initialize Gemini Client: {e}")
        client = None
else:
    client = None

# --- 2. The Simulated MCP Server (External Data Source) ---
class MyCustomMCPServer:
    """
    A class that simulates an MCP server providing tools and data.
    This acts as the agent's gateway to 'external' data.
    """
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
                data_key = call.args.get("data_key")
                if data_key in self.PRIVATE_DATA:
                    result_data = self.PRIVATE_DATA[data_key]
                    # Create the custom ToolResult object
                    return ToolResult(
                        toolName=call.name,
                        invocationId=call.invocationId,
                        result={"status_data": result_data}
                    )
                else:
                    return ToolResult(
                        toolName=call.name,
                        invocationId=call.invocationId,
                        error=f"Error: Data key '{data_key}' not found in internal system."
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
    
    # 🔴 FIX APPLIED HERE: tools argument passed directly, not in a config dict
    chat = client.chats.create(
        model=MODEL_NAME,
        system_instruction="You are a specialized Internal Information Agent. You must use the 'get_internal_status' tool to answer any questions related to company projects, status, or contacts. If a question is external (e.g., 'What is the weather?'), answer directly without using the tool.",
        tools=[server.MCP_TOOL_SPEC] # ⬅️ Correct way to pass tools to client.chats.create
    )
    
    current_prompt = prompt
    response = chat.send_message(current_prompt)
    
    # Multi-turn interaction loop for tool use
    while response.function_calls:
        st.info("Agent is detecting a need for external data...")
        
        # 1. LLM requested a tool call
        st.markdown("**➡️ Tool Call Detected:**")
        
        tool_results_list = []
        for call in response.function_calls:
            st.code(f"Tool: {call.name} | Args: {dict(call.args)}", language="json")
            
            # 2. Execute the simulated MCP server tool
            invocation_id = str(uuid.uuid4())
            # Convert Gemini FunctionCall to custom ToolCall for execution compatibility
            mcp_tool_call = ToolCall(
                name=call.name,
                invocationId=invocation_id,
                args=dict(call.args)
            )
            
            # The tool executes the logic defined in our MyCustomMCPServer
            mcp_result: ToolResult = server.execute_tool(mcp_tool_call)
            
            # Convert custom ToolResult back to Gemini ToolResult format
            gemini_tool_result = types.ToolResult(
                function_name=mcp_result.toolName,
                response={
                    # Gemini expects the JSON response to match the tool specification.
                    "status_data": mcp_result.result.get("status_data") if mcp_result.result else mcp_result.error
                }
            )
            tool_results_list.append(gemini_tool_result)

            st.markdown(f"**⬅️ Tool Result:**")
            st.code(json.dumps(mcp_result.result if mcp_result.result else mcp_result.error, indent=2), language='json')
            
        # 3. Send the tool results back to the LLM
        st.info("Sending tool results back to the Agent for final answer generation...")
        response = chat.send_message(
            current_prompt, # Send original prompt back to chat to retain context
            tool_results=tool_results_list
        )
    
    # 4. Final response
    st.subheader("✅ Final Agent Response")
    st.markdown(response.text)


# --- 4. Streamlit UI ---

st.title("Gemini Agent with Model Context Protocol (MCP) Simulation")

st.markdown("""
<div style="padding: 10px; background-color: #ffe0e0; border-radius: 8px;">
    🚨 **Error Fix:** The `TypeError` was resolved by correcting how the tool specification is passed to the Gemini SDK's `client.chats.create` method. The `tools` argument is now passed directly, as required by the SDK.
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
