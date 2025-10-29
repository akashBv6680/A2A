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
API_KEY = os.environ.get("GEMINI_API_KEY", st.secrets.get("GEMINI_API_KEY", ""))
MODEL_NAME = "gemini-2.5-flash-preview-09-2025" 

st.set_page_config(layout="wide", page_title="Gemini Agent with Custom Data Uploader")

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
    def __init__(self):
        # Data available to 'get_internal_status' tool (static data)
        self.PRIVATE_DATA = {
            "project_status": "The 'Quantum Leap' project is 85% complete. Final testing scheduled for 2026-03-15.",
            "internal_contact": "Lead Engineer: Dr. Anya Sharma (ext: 5891)",
            "security_policy": "All unreleased technical specifications are classified Level-3 and cannot be shared externally."
        }
        # Data populated by the file uploader (dynamic data)
        self.UPLOADED_DATA: Optional[str] = None
    
    # --- New Function: Loads and stores file data (Simulates Document Processing) ---
    def load_file_data(self, file_bytes: bytes, file_name: str) -> str:
        """Processes the uploaded file bytes and stores relevant text."""
        try:
            # For simplicity, we assume text-based files (txt, simple PDF content, etc.)
            # In a real app, you'd use libraries like PyPDF2 or Unstructured for PDFs
            content = file_bytes.decode('utf-8')
            self.UPLOADED_DATA = content
            
            # Simple summarization for the LLM to process
            summary = f"File Content Summary from '{file_name}': The file contains {len(content)} characters. The first 100 characters are: '{content[:100]}...'"
            
            return summary
        except Exception as e:
            return f"Error processing file: {e}"

    # The formal tool specification for the LLM to read (static data tool)
    MCP_INTERNAL_TOOL_SPEC = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="get_internal_status",
                description="Retrieves the current status and key details from the company's internal project management system. Use this to answer questions about projects or contacts.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "data_key": types.Schema(
                            type=types.Type.STRING,
                            description="The specific piece of data to retrieve (e.g., 'project_status', 'internal_contact')."
                        )
                    },
                    required=["data_key"],
                ),
            )
        ]
    )

    # --- New Tool: For uploaded data retrieval ---
    MCP_UPLOAD_TOOL_SPEC = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="get_uploaded_document_data",
                description="Retrieves the content of the currently uploaded file. ONLY use this when the user asks a question ABOUT the file they uploaded (e.g., 'Summarize the document', 'What does the file say about X?').",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "query": types.Schema(
                            type=types.Type.STRING,
                            description="A short, direct summary of what the user wants to know about the uploaded file."
                        )
                    },
                    required=["query"],
                ),
            )
        ]
    )

    def execute_tool(self, call: ToolCall) -> ToolResult:
        """Executes the tool call and returns an MCP ToolResult."""
        if call.name == "get_internal_status":
            # (Execution logic for static data remains the same)
            try:
                requested_key = call.args.get("data_key", "").lower().replace(' ', '_')
                
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
                        error=f"Error: Static Data key '{call.args.get('data_key')}' not found."
                    )
            except Exception as e:
                return ToolResult(
                    toolName=call.name,
                    invocationId=call.invocationId,
                    error=f"Execution error: {str(e)}"
                )
        
        elif call.name == "get_uploaded_document_data":
            # --- New Tool Execution Logic: Retrieve Uploaded Data ---
            if self.UPLOADED_DATA:
                # The prompt is the LLM's query about the document, which we send back
                # This simulates a smarter RAG system that uses the LLM to process the raw text
                # We return the ENTIRE document text for the LLM to reason over
                return ToolResult(
                    toolName=call.name,
                    invocationId=call.invocationId,
                    result={"document_text": self.UPLOADED_DATA}
                )
            else:
                return ToolResult(
                    toolName=call.name,
                    invocationId=call.invocationId,
                    error="No document has been uploaded or processed yet."
                )
        
        else:
            return ToolResult(
                toolName=call.name,
                invocationId=call.invocationId,
                error=f"Tool not implemented: {call.name}"
            )

# --- 3. Agent Workflow Function (Uses generate_content for robustness) ---
def run_mcp_agent_workflow(prompt: str, server: MyCustomMCPServer):
    if not client:
        st.error("Client not initialized. Check API key.")
        return

    st.subheader("🤖 Agent 1: The Multi-Tool Agent")
    
    # 1. FIRST TURN: Send the prompt and enable BOTH tools
    st.info("Agent is processing prompt and checking for tool need...")
    
    # The agent now has access to both the internal and the file processing tool
    tools_list = [server.MCP_INTERNAL_TOOL_SPEC, server.MCP_UPLOAD_TOOL_SPEC]
    
    first_response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[prompt],
        config=types.GenerateContentConfig( 
            system_instruction="You are a specialized Multi-Tool Agent. Use 'get_internal_status' for project/contact questions and 'get_uploaded_document_data' for questions about the uploaded file. If a question is external (e.g., 'What is the weather?'), answer directly.",
            tools=tools_list
        )
    )
    
    # Check if a tool call was made
    if not first_response.function_calls:
        st.subheader("✅ Final Agent Response (No Tool Used)")
        st.markdown(first_response.text)
        return

    # --- Tool Call Loop Initiated ---
    st.info("Agent detected a need for external data. Executing tool...")

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
        
    # 3. SECOND TURN: Send the * entire conversation * plus tool results for the final answer
    st.info("Sending full history and tool results back to the Agent for final answer generation...")

    contents_for_second_turn = [
        types.Content(role="user", parts=[types.Part(text=prompt)]), 
        first_response.candidates[0].content,                            
        types.Content(role="tool", parts=tool_parts_list)                
    ]
    
    final_response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents_for_second_turn,
        config=types.GenerateContentConfig( 
            # Re-pass the system instruction and tools for the second turn
            system_instruction="You are a specialized Multi-Tool Agent. Use 'get_internal_status' for project/contact questions and 'get_uploaded_document_data' for questions about the uploaded file. If a question is external (e.g., 'What is the weather?'), answer directly.",
            tools=tools_list
        )
    )

    # 4. Final response
    st.subheader("✅ Final Agent Response")
    st.markdown(final_response.text)


# --- 4. Streamlit UI and File Upload ---

st.title("Gemini Agent with Model Context Protocol (MCP) and Data Uploader")

# Initialize the simulated MCP server
if 'mcp_server_instance' not in st.session_state:
    st.session_state.mcp_server_instance = MyCustomMCPServer()

mcp_server_instance = st.session_state.mcp_server_instance

st.divider()

# --- File Uploader Widget ---
st.header("Upload Your Document")
uploaded_file = st.file_uploader(
    "Upload a .txt or any text-based file to ask questions about its content:",
    type=["txt", "log"], # Restrict to simple text files for this demo
    key="file_uploader"
)

# Process the file immediately after upload
if uploaded_file is not None and mcp_server_instance.UPLOADED_DATA is None:
    file_bytes = uploaded_file.read()
    file_name = uploaded_file.name
    
    # Call the server's function to process the file
    summary_message = mcp_server_instance.load_file_data(file_bytes, file_name)
    st.success(f"File '{file_name}' loaded successfully! The agent can now use its content.")
    st.caption(summary_message)
    # Clear the uploader for the next session if the file is processed
    uploaded_file = None 
    
elif uploaded_file is None and mcp_server_instance.UPLOADED_DATA is not None:
    st.warning("File processing complete. You can now ask questions about the document!")


st.divider()
st.header("Ask the Agent")

prompt = st.text_input(
    "Ask the Agent a question:",
    placeholder="e.g., 'Summarize the document I uploaded' OR 'What is the status of the Quantum Leap project?'",
    key="prompt_input"
)

if st.button("Run Multi-Tool Agent", type="primary") and prompt:
    if client:
        with st.spinner("Executing Agent Workflow..."):
            run_mcp_agent_workflow(prompt, mcp_server_instance)
    else:
        st.error("Cannot run. Gemini Client is not initialized.")
