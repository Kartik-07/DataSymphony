# main.py
import os
import subprocess
import sys
import traceback
import tempfile
import logging
import base64
from fastapi import FastAPI, Request, HTTPException, status
from pydantic import BaseModel
# Importing uvicorn isn't strictly necessary for the code to run when started by uvicorn command

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure a directory for potential plot outputs within the container
ARTIFACT_DIR = tempfile.mkdtemp()
logger.info(f"Artifact directory created at: {ARTIFACT_DIR}")

# --- Pydantic Model for Request Body ---
class CodePayload(BaseModel):
    code: str

# --- FastAPI Endpoint ---
@app.post("/execute")
async def execute_code(payload: CodePayload):
    """
    Receives Python code via POST request body and executes it.
    """
    code = payload.code
    logger.info("Received code for execution.")

    # Very basic security check (can be expanded)
    # Avoid allowing imports that could be used maliciously in this context
    if "import os" in code or "import subprocess" in code or "import sys" in code:
         logger.warning("Execution attempt with potentially unsafe imports blocked.")
         raise HTTPException(
             status_code=status.HTTP_403_FORBIDDEN,
             detail="Code contains potentially unsafe imports (os, subprocess, sys)."
         )

    stdout_result = ""
    stderr_result = ""
    plot_artifact_base64 = None
    script_path = None # Keep track of the temp file path for logging

    try:
        # Use a temporary file to write the code for execution
        # delete=False is important here so subprocess can access it before the 'with' block closes it
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_code_file:
            script_path = tmp_code_file.name # Store path for logging/cleanup
            # Prepend necessary imports to the user's code
            tmp_code_file.write("import pandas as pd\n")
            tmp_code_file.write("import numpy as np\n")
            tmp_code_file.write("import matplotlib\n")
            tmp_code_file.write("matplotlib.use('Agg') # Use non-interactive backend suitable for scripts/servers\n")
            tmp_code_file.write("import matplotlib.pyplot as plt\n")
            tmp_code_file.write("import scipy\n")
            # Ensure plots are saved to the correct directory within the container
            tmp_code_file.write(f"import os\nos.chdir('{ARTIFACT_DIR}')\n")
            tmp_code_file.write("\n# --- User Code Start ---\n")
            tmp_code_file.write(code)
            tmp_code_file.write("\n# --- User Code End ---\n")
            tmp_code_file.flush() # Ensure all content is written to the file

        logger.info(f"Executing code from temporary file: {script_path}")

        # Execute the temporary file using the python interpreter in a subprocess
        process = subprocess.run(
            [sys.executable, script_path], # Run the script using the same Python interpreter
            capture_output=True,           # Capture stdout and stderr
            text=True,                     # Decode output as text
            timeout=60,                    # Set a 60-second timeout to prevent long-running code
            check=False                    # Do not raise exception on non-zero exit code, capture stderr instead
        )

        stdout_result = process.stdout
        stderr_result = process.stderr

        logger.info(f"Execution finished. Return code: {process.returncode}")
        if stdout_result: logger.debug(f"STDOUT:\n{stdout_result}")
        if stderr_result: logger.warning(f"STDERR:\n{stderr_result}")

        # Optional: Check for a standard plot artifact (e.g., 'plot.png')
        plot_path = os.path.join(ARTIFACT_DIR, 'plot.png')
        if os.path.exists(plot_path):
             logger.info("Plot artifact 'plot.png' found. Encoding...")
             with open(plot_path, 'rb') as f:
                 plot_artifact_base64 = base64.b64encode(f.read()).decode('utf-8')
             try:
                 os.remove(plot_path) # Clean up the plot file
             except OSError as e:
                 logger.error(f"Error deleting plot file {plot_path}: {e}")


        # Return the results
        return {
            "stdout": stdout_result,
            "stderr": stderr_result,
            "plot_png_base64": plot_artifact_base64, # Include base64 plot if found
            "execution_successful": process.returncode == 0
        }

    except subprocess.TimeoutExpired:
        logger.error("Code execution timed out.")
        # If timeout occurs, raise an HTTP exception
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail={ # Include partial output if available
                "message": "Code execution timed out after 60 seconds.",
                "stdout": stdout_result,
                "stderr": stderr_result,
            }
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred during code execution: {e}\n{traceback.format_exc()}")
        # For any other errors, raise an internal server error
        raise HTTPException(
             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
             detail={ # Include partial output if available
                "message": f"Internal error during code execution: {e}",
                "stdout": stdout_result,
                "stderr": stderr_result,
             }
         )
    finally:
        # Crucial: Clean up the temporary script file regardless of success or failure
        if script_path and os.path.exists(script_path):
            try:
                os.remove(script_path)
                logger.debug(f"Deleted temporary script: {script_path}")
            except OSError as e:
                logger.error(f"Error deleting temporary script {script_path}: {e}")

# Note: No `if __name__ == '__main__':` block is needed.
# Uvicorn will be called via the Docker CMD instruction to run the 'app' instance.