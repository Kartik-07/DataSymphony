// src/api/ragApi.js

import axios from 'axios';

// Make sure this matches the address where your FastAPI backend is running
const API_BASE_URL = 'http://localhost:8000'; // Or your actual backend URL

// Create an Axios instance
const apiClient = axios.create({
  baseURL: API_BASE_URL,
});

// --- Axios Request Interceptor ---
// This automatically adds the Authorization header to requests if a token exists
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('authToken'); // Get token from localStorage
    if (token) {
      config.headers['Authorization'] = `Bearer ${token}`;
      // console.debug("Interceptor: Added Auth header"); // Uncomment for debugging
    }
    // console.debug("Interceptor: Request config:", config); // Uncomment for debugging
    return config;
  },
  (error) => {
    console.error("Interceptor Request Error:", error);
    return Promise.reject(error); // Pass the error along
  }
);

// --- Helper for Error Handling ---
// This helper centralizes error logging and message generation
const handleApiError = (error, context) => {
    console.error(`Error ${context}:`, error);
    let errorMessage = `Failed to ${context}.`;

    if (error.response) {
      // Server responded with a status code outside 2xx range
      console.error('Error Response:', {
          status: error.response.status,
          statusText: error.response.statusText,
          data: error.response.data // Contains backend error details (e.g., 'detail')
      });
      // Use backend's detail message if available, otherwise fallback
      errorMessage = `Server error during ${context}: ${error.response.status} ${error.response.data?.detail || error.response.statusText || 'Unknown server error'}`;
      // Specific check for 401 Unauthorized
      if (error.response.status === 401) {
          errorMessage = `Authentication failed during ${context}. Please check credentials or token validity.`;
          // Optionally: trigger logout or token refresh logic here
          // e.g., localStorage.removeItem('authToken'); window.location.reload();
      }
    } else if (error.request) {
      // Request was made but no response received (network error, server down)
      errorMessage = `No response received from server during ${context}. Is the backend running and accessible at ${API_BASE_URL}?`;
    } else {
      // Something else happened in setting up the request (e.g., code error before sending)
      errorMessage = `Request setup error during ${context}: ${error.message}`;
    }
    // Throwing an error here ensures Promise rejection, which can be caught in components
    throw new Error(errorMessage);
};


// --- Authentication Functions ---

/**
 * Registers a new user.
 * @param {string} email
 * @param {string} password
 * @returns {Promise<object>} User data (excluding password) on success.
 * @throws {Error} If registration fails.
 */
export const registerUser = async (email, password) => {
    try {
        console.log(`Registering user: ${email}`);
        // Use the apiClient instance which includes the interceptor (though not needed for register)
        const response = await apiClient.post('/register', { email, password });
        console.log('Registration successful:', response.data);
        return response.data; // Expected: { email: "..." }
    } catch (error) {
        handleApiError(error, `registering user ${email}`);
    }
};

/**
 * Logs in a user and retrieves an access token.
 * @param {string} email
 * @param {string} password
 * @returns {Promise<string>} Access token on success.
 * @throws {Error} If login fails.
 */
export const loginUser = async (email, password) => {
    try {
        console.log(`Logging in user: ${email}`);
        // Use standard form data for OAuth2PasswordRequestForm
        // Axios can send URLSearchParams directly
        const params = new URLSearchParams();
        params.append('username', email); // FastAPI expects 'username' for OAuth2 form
        params.append('password', password);

        const response = await apiClient.post('/token', params, {
             headers: { 'Content-Type': 'application/x-www-form-urlencoded' } // Required header for form data
        });
        console.log('Login successful');
        // Expecting { access_token: "...", token_type: "bearer" }
        if (response.data && response.data.access_token) {
            return response.data.access_token;
        } else {
            throw new Error("Access token not found in login response.");
        }
    } catch (error) {
         handleApiError(error, `logging in user ${email}`);
    }
};

/**
 * Fetches information about the currently authenticated user.
 * Requires a valid token to be stored.
 * @returns {Promise<object>} User data.
 * @throws {Error} If token is invalid or request fails.
 */
export const fetchCurrentUser = async () => {
    try {
        console.log(`Fetching current user info`);
        // Interceptor will add the token
        const response = await apiClient.get('/users/me');
        console.log('Current user info:', response.data);
        return response.data; // Expected: { email: "..." }
    } catch (error) {
         handleApiError(error, `fetching current user info`);
         // NOTE: A 401 error here likely means the stored token is invalid/expired
    }
};


// --- RAG & Conversation Functions ---

/**
 * Sends a query to the RAG API backend.
 * Requires a valid token (added by interceptor).
 * @param {string} query - The user's question.
 * @param {string | null} conversationId - ID of the current conversation for context.
 * @param {string | null} [tempFileId=null] - Optional ID of a temporary file for context.
 * @returns {Promise<object>} - API response object {answer, sources?, suggestions?}.
 * @throws {Error} If API call fails.
 */
export const fetchQueryResponse = async (query, conversationId, tempFileId = null) => {
  try {
    const payload = {
      query: query,
      conversation_id: conversationId,
      temp_file_id: tempFileId // Include the temp file ID
    };
    console.log(`Sending query to /query:`, payload);
    // Interceptor adds token header
    const response = await apiClient.post(`/query`, payload); // Send payload as JSON
    console.log('Received query response:', response.data);
    return response.data;
  } catch (error) {
    handleApiError(error, "fetching query response");
  }
};

/**
 * Uploads a file for PERMANENT ingestion via background task.
 * Requires a valid token (added by interceptor).
 * @param {File} file - The file object to upload.
 * @returns {Promise<object>} - API response object { status, message, filename }.
 * @throws {Error} If upload fails.
 */
export const uploadFileForIngestion = async (file) => {
    if (!file) {
        throw new Error("No file selected for upload.");
    }
    const formData = new FormData();
    formData.append('file', file);

    try {
        console.log(`Uploading file to /ingest_file for permanent ingestion:`, file.name);
        // Interceptor adds token header
        const response = await apiClient.post(`/ingest_file`, formData, {
            // Axios sets Content-Type correctly for FormData automatically
            // headers: { 'Content-Type': 'multipart/form-data' }, // Usually not needed
        });
        console.log('Permanent ingestion upload response:', response.data);
        // Expecting { status: "scheduled", message: "...", filename: "..." }
        return response.data;
    } catch (error) {
        handleApiError(error, "uploading file for permanent ingestion");
    }
};

/**
 * Fetches the list of conversations from the backend.
 * Requires a valid token (added by interceptor).
 * @returns {Promise<Array<object>>} - Array of conversation list item objects.
 * @throws {Error} If API call fails.
 */
export const fetchConversations = async () => {
    try {
        console.log(`Fetching conversation list from /conversations`);
        // Interceptor adds token header
        const response = await apiClient.get(`/conversations`);
        console.log('Received conversation list:', response.data);
        return response.data || []; // Return empty array if null/undefined
    } catch (error) {
         handleApiError(error, "fetching conversation list");
         return []; // Return empty array on error to prevent crashes
    }
};

/**
 * Fetches the details of a specific conversation.
 * Requires a valid token (added by interceptor).
 * @param {string} conversationId - The unique identifier of the conversation.
 * @returns {Promise<object>} - Conversation data object { id, title, created_at, messages }.
 * @throws {Error} If API call fails or conversation not found.
 */
export const fetchConversation = async (conversationId) => {
    if (!conversationId) {
        throw new Error("Conversation ID is required to fetch details.");
    }
    try {
        console.log(`Fetching conversation details for ID: ${conversationId}`);
        // Interceptor adds token header
        const response = await apiClient.get(`/conversations/${conversationId}`);
        console.log('Received conversation details:', response.data);
        return response.data;
    } catch (error) {
         handleApiError(error, `fetching conversation details for ID ${conversationId}`);
    }
};

/**
 * Saves or updates a conversation on the backend.
 * Requires a valid token (added by interceptor).
 * @param {object} conversationData - { conversation_id?, title?, messages }
 * @returns {Promise<object>} - Backend response { conversation_id, title, message }.
 * @throws {Error} If API call fails.
 */
export const saveConversation = async (conversationData) => {
    if (!conversationData || !Array.isArray(conversationData.messages)) {
        throw new Error("Invalid conversation data provided for saving.");
    }
    try {
        const conversationId = conversationData.conversation_id;
        const url = `/conversations`; // Single endpoint handles create/update

        console.log(`Saving conversation (ID: ${conversationId || 'New'})`);
        // Interceptor adds token header
        const response = await apiClient.post(url, conversationData); // Send the whole object as JSON
        console.log('Save conversation response:', response.data);
        return response.data;
    } catch (error) {
        handleApiError(error, `saving conversation (ID: ${conversationData.conversation_id || 'New'})`);
    }
};

// --- NEW: Temporary File Upload Function ---
/**
 * Uploads a temporary file for query context.
 * Requires a valid token (added by interceptor).
 * @param {File} file - The file object to upload.
 * @returns {Promise<object>} - The response object { file_id, filename }.
 * @throws {Error} If upload fails.
 */
export const uploadTemporaryFile = async (file) => {
    if (!file) throw new Error("File object is required for temporary upload.");

    const formData = new FormData();
    formData.append('file', file); // Backend expects the file under the key 'file'

    try {
        console.log(`Uploading temporary file to /upload_temp: ${file.name}`);
        // Interceptor adds token header
        const response = await apiClient.post(`/upload_temp`, formData, {
            // Axios sets Content-Type correctly for FormData automatically
        });
        console.log('Temporary file upload response:', response.data);
        // Expecting { file_id: "...", filename: "..." }
        if (response.data && response.data.file_id) {
            return response.data;
        } else {
            throw new Error("File ID not found in temporary upload response.");
        }
    } catch (error) {
        handleApiError(error, `uploading temporary file ${file.name}`);
    }
};


// Optional: Function to check API status (doesn't need auth)
export const checkApiStatus = async () => {
    try {
        // Use base axios instance for non-authenticated endpoints if preferred,
        // or just use apiClient if the endpoint doesn't require auth anyway.
        const response = await axios.get(`${API_BASE_URL}/`); // Direct axios call
        return response.data;
    } catch (error) {
        console.error('Error checking API status:', error);
        // Provide a structured error response
        return { status: 'error', message: `Could not connect to API at ${API_BASE_URL}. Is it running?` };
    }
}
