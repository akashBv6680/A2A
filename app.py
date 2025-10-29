import streamlit as st
import os
import json
import uuid
from google import genai
from google.genai import types
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

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
                # Normalize the LLM's requested key to match our internal data keys
                requested_key = call.args.get("data_key", "").lower().replace(' ', '_')
                
                # Check for direct matches or common variations
                if requested_key == "project_status" or "quantum_leap" in requested_key:
                    data_key = "project_status"
                elif requested_key == "internal_contact" or "contact" in requested_key:
                    data_key = "internal_contact"
                elif requested_key == "security_policy" or "policy" in requested_key:
                    data_key = "security_policy"
                else:
                    data_key = None # No match
                
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
    
    # Passing both system_instruction and tools within the explicit config object
    chat = client.chats.create(
        model=MODEL_NAME,
        config=types.GenerateContentConfig( 
            system_instruction="You are a specialized Internal Information Agent. You must use the 'get_internal_status' tool to answer any questions related to company projects, status, or contacts. If a question is external (e.g., 'What is the weather?'), answer directly without using the tool.",
            tools=[server.MCP_TOOL_SPEC]
        )
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
            mcp_tool_call = ToolCall(
                name=call.name,
                invocationId=invocation_id,
                args=dict(call.args)
            )
            
            # The tool executes the logic defined in our MyCustomMCPServer
            mcp_result: ToolResult = server.execute_tool(mcp_tool_call)
            
            if mcp_result.result:
                result_content = mcp_result.result
            else:
                result_content = {"error_message": mcp_result.error}
            
            # 🛑 FINAL FIX APPLIED HERE: Creating the required FunctionResponse structure
            # This is the expected structure for tool results in the chat API:
            gemini_tool_result = types.Part.from_function_response( # Use the helper function for correct typing
                name=mcp_result.toolName,
                response=result_content # The content of the result
            )
            
            tool_results_list.append(gemini_tool_result)

            st.markdown(f"**⬅️ Tool Result:**")
            st.code(json.dumps(result_content, indent=2), language='json')
            
        # 3. Send the tool results back to the LLM
        st.info("Sending tool results back to the Agent for final answer generation...")
        
        # We must construct a list of Contents, where each Content has the tool results
        # When passing tool_results to send_message, we pass a list of Parts, not a list of dictionaries.
        response = chat.send_message(
            contents=[
                types.Content(
                    parts=tool_results_list,
                    role='tool' # Must specify the role is 'tool' for function responses
                )
            ]
        )
    
    # 4. Final response
    st.subheader("✅ Final Agent Response")
    st.markdown(response.text)


# --- 4. Streamlit UI ---

st.title("Gemini Agent with Model Context Protocol (MCP) Simulation")

st.markdown("""
<div style="padding: 10px; background-color: #d1e7dd; border-radius: 8px;">
    🎉 **Final Error Fix:** The last `TypeError` was resolved by ensuring the tool results are sent back using the **correct `types.Content` object structure with the `role='tool'`** and using `types.Part.from_function_response` to create the parts. This complex, multi-step process should now successfully complete the Agent-to-Model communication cycle!
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
