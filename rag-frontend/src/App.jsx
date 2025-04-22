// src/App.jsx

import React, { useState, useEffect, useRef, useCallback } from 'react';
// --- NEW: Import icons ---
import { FaPaperclip, FaPaperPlane, FaTimesCircle } from 'react-icons/fa'; // Example using react-icons

// API function imports
import {
    fetchQueryResponse,
    fetchConversations,
    fetchConversation,
    saveConversation,
    registerUser,
    loginUser,
    fetchCurrentUser,
    // --- NEW: Add API function for temporary upload ---
    uploadTemporaryFile // Assuming this exists in ragApi.js
    // uploadFileForIngestion // Keep if needed for permanent ingestion elsewhere
} from './api/ragApi';
import './App.css'; // Ensure CSS is linked

// --- Constants for File Upload ---
const ALLOWED_MIME_TYPES = [
  'text/plain', // .txt
  'application/pdf', // .pdf
  'application/vnd.ms-excel', // .xls (older excel)
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', // .xlsx
  'text/csv', // .csv
  'application/msword', // .doc (older word)
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document', // .docx
];
const MAX_FILE_SIZE_MB = 25; // Example limit: 25MB


// --- Icon/Helper Components (Unchanged) ---
const AiIcon = () => ( <div className="ai-icon-container"> <img src="/Chatbot-svg.svg" alt="AI Icon" width="50" height="50" /> </div> );
const UserIcon = () => ( <div className="user-icon-container"> KV </div> ); // Consider making dynamic based on logged-in user
const FileIcon = () => ( <svg viewBox="0 0 24 24" width="12" height="12" fill="currentColor" style={{ marginRight: '4px', flexShrink: 0 }}> <path d="M14 2H6c-1.1 0-1.99.9-1.99 2L4 20c0 1.1.89 2 1.99 2H18c1.1 0 2-.9 2-2V8l-6-6zM6 20V4h7v5h5v11H6z"/> </svg> );

// --- SourceItem Component (Unchanged) ---
const SourceItem = ({ source }) => {
    const sourceName = source?.source || 'Unknown Source';
    const sourceType = source?.type || 'unknown';
    const isInternal = sourceType === 'internal';
    const isPlot = sourceType === 'plot_png_base64'; // Check if it's a plot

    // Skip rendering if it's a plot source (handled directly in Message)
    if (isPlot) {
        return null;
    }

    const itemClass = `source-item ${isInternal ? 'source-item-internal' : ''}`;

    return (
        <li className={itemClass} title={sourceName}>
            {!isInternal && <FileIcon />}
            <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {sourceName}
            </span>
        </li>
    );
};


// --- Sources Component (Unchanged) ---
const Sources = ({ sources }) => {
    if (!Array.isArray(sources) || sources.length === 0) { return null; }

    // Filter out internal knowledge and plot sources for the list display
    const displaySources = sources.filter(source => {
        const isInternal = source?.type === 'internal';
        const isPlot = source?.type === 'plot_png_base64';
        return !isInternal && !isPlot; // Only show non-internal, non-plot sources
    });

    // Handle case where only internal knowledge or only plots were present
    const internalSource = sources.find(source => source?.type === 'internal');
    if (displaySources.length === 0 && internalSource) {
         return (
             <div className="sources-section">
                 <h4 className="sources-title">SOURCE:</h4>
                 <ul className="source-list">
                     {/* Render only the internal source indicator */}
                     <li className="source-item source-item-internal" title="Internal Knowledge">
                        <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                            Internal Knowledge
                        </span>
                     </li>
                 </ul>
             </div>
         );
    }

    // If no displayable sources remain (and no internal source either), return null
    if (displaySources.length === 0) { return null; }

    return (
        <div className="sources-section">
            <h4 className="sources-title">SOURCES:</h4>
            <ul className="source-list">
                {displaySources.map((source, index) => (
                    // Use identifier if available, otherwise index
                    <SourceItem key={source.identifier || `source-${index}`} source={source} />
                ))}
            </ul>
        </div>
    );
};


// ChatHeader Component (Unchanged)
const ChatHeader = ({ title }) => (
    <div className="chat-header">
        <div>
            <h1>{title}</h1>
            <span>AI-powered assistant</span>
        </div>
    </div>
);

// --- Message Component (Unchanged) ---
const Message = ({ message }) => {
    const { sender, text, sources, error } = message;
    const messageClass = sender === 'user' ? 'user' : 'ai';
    const bubbleClass = error ? 'error' : messageClass;

    // --- Find the plot source (if any) ---
    const plotSource = sources?.find(source => source?.type === 'plot_png_base64' && source.data);

    return (
        <div className={`message-container ${messageClass}`}>
            {sender === 'ai' && <AiIcon />}
            {sender === 'user' && <UserIcon />}
            <div className="message-content-wrapper">
                {/* Text Bubble */}
                <div className={`message-bubble ${bubbleClass}`}>
                    <span style={{ whiteSpace: 'pre-wrap' }}>{text}</span>
                </div>

                {/* --- Render Plot directly if found --- */}
                {sender === 'ai' && plotSource && (
                    <div className="plot-display" style={{ /* Use inline or CSS */ alignSelf: 'flex-start', marginTop: '8px'}}>
                         <img
                             src={`data:image/png;base64,${plotSource.data}`}
                             alt={plotSource.source || "Generated Plot"}
                             style={{ maxWidth: '100%', height: 'auto', display: 'block', border: '1px solid #eee', borderRadius: '4px' }}
                         />
                    </div>
                )}
                {/* --- END Plot Rendering --- */}

                {/* Render Sources list (always pass all sources, SourceItem handles filtering) */}
                {sender === 'ai' && !error && <Sources sources={sources} />}
            </div>
        </div>
    );
};


// MessageList Component (Unchanged)
const MessageList = ({ messages }) => {
    const messagesEndRef = useRef(null);
    useEffect(() => {
        const timer = setTimeout(() => {
            messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
        }, 100); // Short delay to allow rendering
        return () => clearTimeout(timer);
    }, [messages]); // Dependency on messages array
    return (
        <div className="message-list">
            {messages.map((msg, index) => (
                <Message key={msg.timestamp || index} message={msg} /> // Use timestamp or index as key
            ))}
            <div ref={messagesEndRef} />
        </div>
    );
};

// InputArea Component (Simplified - Renders only the textarea)
const InputArea = ({ inputValue, onInputChange, isLoading, textareaRef, onKeyDown }) => {
    // Auto-resize logic remains
    useEffect(() => {
        const textarea = textareaRef.current;
        if (textarea) {
            textarea.style.height = 'auto'; // Reset height
            const scrollHeight = textarea.scrollHeight;
            // Set height based on content, consider max-height from CSS
            textarea.style.height = `${scrollHeight}px`;
        }
    }, [inputValue, textareaRef]); // Depend on inputValue and ref

    return (
        <textarea
            ref={textareaRef}
            className="chat-textarea" // Added specific class for textarea
            value={inputValue}
            onChange={onInputChange}
            onKeyDown={onKeyDown} // Pass keydown handler from App
            placeholder="Ask Symphony anything..."
            rows="1"
            disabled={isLoading}
            aria-label="Chat input"
        />
    );
};


// SuggestedActions Component (Unchanged)
const SuggestedActions = ({ actions, onActionClick, isLoading }) => {
    if (!Array.isArray(actions) || actions.length === 0) { return null; }
    return (
        <div className="suggested-actions">
            <span className="try-asking-label">Try asking:</span>
            {actions.map(action => (
                <button key={action} onClick={() => onActionClick(action)} disabled={isLoading}>
                    {action}
                </button>
            ))}
        </div>
    );
}

// Sidebar Component (Unchanged)
const Sidebar = ({ conversationList, activeConversationId, onSelectConversation, onNewChat, onLogout, isAuthenticated }) => {
    const formatTimestamp = (isoTimestamp) => {
        if (!isoTimestamp) return '';
        try {
            const date = new Date(isoTimestamp);
            // Example format: Apr 16, 1:09 AM
            return date.toLocaleDateString([], { month: 'short', day: 'numeric' }) + ', ' +
                   date.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit', hour12: true });
        } catch { return ''; } // Handle potential invalid date strings
    }

    return (
        <div className="sidebar">
            <div className="sidebar-conversation-list">
                <h2 className="sidebar-title">Conversations</h2>
                {conversationList.length === 0 && isAuthenticated && <p className="sidebar-empty">No past conversations.</p>}
                {!isAuthenticated && <p className="sidebar-empty">Please log in to see conversations.</p>}
                {isAuthenticated && conversationList.map(conv => (
                    <div
                        key={conv.id}
                        className={`sidebar-item ${conv.id === activeConversationId ? 'active' : ''}`}
                        onClick={() => onSelectConversation(conv.id)}
                        title={conv.title}
                        role="button"
                        tabIndex={0} // Make it focusable
                        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') onSelectConversation(conv.id);}} // Keyboard accessible
                    >
                        <span className="sidebar-item-title">{conv.title}</span>
                        {/* Display the most recent message timestamp for sorting */}
                        {conv.timestamp && <span className="sidebar-item-time">{formatTimestamp(conv.timestamp)}</span>}
                    </div>
                ))}
            </div>
            {isAuthenticated && (
                <>
                    <button className="new-chat-button" onClick={onNewChat}> + New Chat </button>
                    <button className="logout-button" onClick={onLogout}> Logout </button>
                </>
            )}
        </div>
    );
};

// AuthForm Component (Unchanged)
const AuthForm = ({ isLoginMode, onSubmit, error, onToggleMode, isLoading }) => {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');

    const handleSubmit = (e) => {
        e.preventDefault();
        if (!isLoading) { onSubmit(email, password); }
    };

    const title = isLoginMode ? 'Login' : 'Register';
    const subtitle = isLoginMode ? 'Welcome back! Please enter your email and password.' : 'Create your account.';
    const buttonText = isLoginMode ? 'Log In' : 'Register';
    const toggleText = isLoginMode ? 'Need an account? Register' : 'Already have an account? Login';

    return (
        <div className="auth-page-wrapper">
            <div className="auth-background-container">
                <div className="auth-card">
                    <div className="auth-left-column">
                        <img
                            src="/left_login_image.png" // Verify path
                            alt="Login visual"
                            className="auth-left-image"
                        />
                    </div>
                    <div className="auth-right-column">
                        <div className="auth-logo-group">
                             <img
                                 src="/Brand mark - transparent - png.png" // Verify path
                                 alt="DataSymphony Logo"
                                 className="auth-logo"
                             />
                             <span className="auth-app-name">DATA SYMPHONY</span>
                        </div>
                        <h3>{title}</h3>
                        <p className="auth-subtitle">{subtitle}</p>
                        <form onSubmit={handleSubmit} className="auth-actual-form">
                            <div className="form-group">
                                <label htmlFor="email" className="auth-label">Email ID</label>
                                <input
                                    type="email"
                                    id="email"
                                    className="auth-input"
                                    value={email}
                                    onChange={(e) => setEmail(e.target.value)}
                                    required
                                    autoComplete="email"
                                    placeholder="Enter your email"
                                    disabled={isLoading}
                                />
                            </div>
                            <div className="form-group">
                                <div className="password-label-group">
                                    <label htmlFor="password" className="auth-label">Password</label>
                                    {isLoginMode && <a href="#" className="forgot-password-link">Forgot Password?</a>}
                                </div>
                                <input
                                    type="password"
                                    id="password"
                                    className="auth-input"
                                    value={password}
                                    onChange={(e) => setPassword(e.target.value)}
                                    required
                                    minLength={isLoginMode ? undefined : 8} // Require min length only for registration
                                    autoComplete={isLoginMode ? "current-password" : "new-password"}
                                    placeholder="Enter your password"
                                    disabled={isLoading}
                                />
                            </div>
                            {error && <p className="auth-error">{error}</p>}
                            <button type="submit" className="auth-button" disabled={isLoading}>
                                {isLoading ? 'Processing...' : buttonText}
                            </button>
                        </form>
                         <button onClick={onToggleMode} className="toggle-auth-button" disabled={isLoading}>
                             {toggleText}
                         </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

// --- Main Application Component Definition ---
function App() {
    // --- State Variables ---
    const [messages, setMessages] = useState([]);
    const [inputValue, setInputValue] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [conversationList, setConversationList] = useState([]);
    const [activeConversationId, setActiveConversationId] = useState(null);
    const [currentTitle, setCurrentTitle] = useState("Chat with Symphony");
    const [suggestedActions, setSuggestedActions] = useState([]);
    const [isSaving, setIsSaving] = useState(false);
    const [isAuthenticated, setIsAuthenticated] = useState(false);
    const [token, setToken] = useState(localStorage.getItem('authToken') || null);
    const [authError, setAuthError] = useState(null);
    const [showLogin, setShowLogin] = useState(true);
    const [isAuthLoading, setIsAuthLoading] = useState(true); // Start true until token checked
    const [currentUserEmail, setCurrentUserEmail] = useState('');
    const saveTimeoutRef = useRef(null);
    const textareaRef = useRef(null); // Ref for the textarea

    // --- State for File Uploads (Unchanged) ---
    const [selectedFile, setSelectedFile] = useState(null);
    const [uploadedFileId, setUploadedFileId] = useState(null);
    const [uploadedFileName, setUploadedFileName] = useState('');
    const [isUploading, setIsUploading] = useState(false);
    const [uploadError, setUploadError] = useState(null);
    const fileInputRef = useRef(null);
    // ------------------------------------

    // --- Authentication Handlers (Unchanged)---
    const handleLogin = async (email, password) => {
        setAuthError(null);
        setIsAuthLoading(true);
        try {
            const receivedToken = await loginUser(email, password);
            localStorage.setItem('authToken', receivedToken);
            setToken(receivedToken);
            setIsAuthenticated(true);
            await loadInitialConversations(receivedToken); // Load convos after successful login
        } catch (error) {
            setAuthError(error.message || "Login failed.");
            setIsAuthenticated(false);
        } finally {
            setIsAuthLoading(false);
        }
    };

    const handleRegister = async (email, password) => {
        setAuthError(null);
        setIsAuthLoading(true);
        try {
            await registerUser(email, password);
            // Attempt to auto-login after successful registration
            await handleLogin(email, password);
        } catch (error) {
            // If registration failed, show specific error
            setAuthError(error.message || "Registration failed.");
            setIsAuthenticated(false); // Ensure not authenticated
            setIsAuthLoading(false); // Stop loading indicator
        }
        // No finally here, handleLogin's finally will set loading false if it runs
    };

    const handleLogout = () => {
        localStorage.removeItem('authToken');
        setToken(null);
        setIsAuthenticated(false);
        setCurrentUserEmail('');
        handleNewChat(); // Reset chat state
        setConversationList([]); // Clear conversation list
        setShowLogin(true); // Show login form
        setAuthError(null); // Clear any previous auth errors
        // Clear file upload state on logout
        setSelectedFile(null);
        setUploadedFileId(null);
        setUploadedFileName('');
        setUploadError(null);
        console.log("User logged out.");
    };

    const toggleAuthMode = () => {
        setShowLogin(!showLogin);
        setAuthError(null); // Clear errors when toggling
    };

    // --- Conversation Loading & Management (Unchanged) ---
    const loadInitialConversations = useCallback(async (currentToken) => {
        if (!currentToken) {
            setIsAuthenticated(false);
            setIsAuthLoading(false);
            return;
        }
        setIsAuthLoading(true);
        setIsLoading(true); // Show loading for convo list too
        try {
            const user = await fetchCurrentUser(); // Verifies token and gets user info
            if (user && user.email) {
                setIsAuthenticated(true);
                setCurrentUserEmail(user.email);
                const fetchedList = await fetchConversations();
                if (Array.isArray(fetchedList)) {
                    // Sort by most recent first (using timestamp from list item)
                    fetchedList.sort((a, b) => (new Date(b.timestamp || 0)) - (new Date(a.timestamp || 0)));
                    setConversationList(fetchedList);
                    if (fetchedList.length > 0) {
                        // Automatically select the most recent conversation
                        await handleSelectConversation(fetchedList[0].id);
                    } else {
                        // No conversations, start a new chat state
                        handleNewChat();
                    }
                } else {
                    // Handle case where fetchConversations doesn't return array
                    console.warn("fetchConversations did not return an array:", fetchedList);
                    setConversationList([]);
                    handleNewChat();
                }
            } else {
                 throw new Error("Token validation failed or user email not found."); // Or handle gracefully
            }
        } catch (error) {
            console.error("Initial load/auth failed:", error);
            handleLogout(); // Force logout on error
            setAuthError("Session invalid or expired. Please log in again.");
        } finally {
            setIsAuthLoading(false);
            setIsLoading(false); // Ensure loading stops
        }
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []); // handleSelectConversation and handleNewChat need to be stable (useCallback or defined outside)

    const handleInputChange = (event) => { setInputValue(event.target.value); };

    const handleSuggestedAction = (actionText) => {
        if (!isLoading && isAuthenticated) {
            // Set input value and trigger send
            setInputValue(actionText);
            // Need to use a slight delay or useEffect to trigger send after state update
            // Or pass actionText directly to handleSend
            handleSend(actionText);
        }
    };

    // Debounced save function (Unchanged)
    const triggerSaveConversation = useCallback(() => {
        // Only save if authenticated, have an active ID, and messages exist
        if (!isAuthenticated || isSaving || messages.length === 0 || !activeConversationId) {
            return;
        }

        if (saveTimeoutRef.current) {
            clearTimeout(saveTimeoutRef.current); // Clear previous timeout
        }

        // Set a new timeout
        saveTimeoutRef.current = setTimeout(async () => {
            setIsSaving(true);
            // Create the data payload for saving
            const conversationData = {
                conversation_id: activeConversationId,
                title: currentTitle, // Send current title
                messages: messages // Send current messages array
            };
            try {
                console.log(`Saving convo (debounced): ${activeConversationId}`);
                const saveResponse = await saveConversation(conversationData);
                console.log("Save response (debounced):", saveResponse);

                // Update conversation list title/timestamp if changed or missing
                const listNeedsUpdate = (saveResponse?.title && saveResponse.title !== currentTitle) || !conversationList.find(c => c.id === activeConversationId)?.timestamp;

                if(listNeedsUpdate) {
                    // Use functional update form for setConversationList
                    setConversationList(prevList => prevList.map(conv =>
                        conv.id === activeConversationId
                            ? { ...conv, title: saveResponse.title || currentTitle, timestamp: new Date().toISOString() } // Update title and timestamp
                            : conv
                    ).sort((a, b) => (new Date(b.timestamp || 0)) - (new Date(a.timestamp || 0)))); // Re-sort

                    // Update current title state if it changed in the backend response
                    if (saveResponse?.title && saveResponse.title !== currentTitle) {
                        setCurrentTitle(saveResponse.title);
                    }
                }
            } catch (error) {
                console.error("Failed to save conversation (debounced):", error);
                // Optionally show a non-blocking error to the user
            } finally {
                setIsSaving(false); // Reset saving flag
                saveTimeoutRef.current = null; // Clear the timeout ref
            }
        }, 1500); // Debounce interval (e.g., 1.5 seconds)
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [activeConversationId, messages, currentTitle, isAuthenticated]); // Dependencies for the save logic


    // --- File Upload Handlers (Unchanged) ---
    const handleFileChange = (event) => {
        const file = event.target.files?.[0];
        if (!file) return;

        setUploadError(null); // Clear previous errors

        // Validate file type (using MIME type)
        if (!ALLOWED_MIME_TYPES.includes(file.type)) {
            const allowedExts = ".txt, .pdf, .csv, .xlsx, .docx"; // Show common extensions
            setUploadError(`Invalid file type. Allowed: ${allowedExts}`);
            setSelectedFile(null);
            setUploadedFileName('');
            setUploadedFileId(null);
            if(fileInputRef.current) fileInputRef.current.value = ''; // Clear input
            return;
        }

        // Validate file size
        if (file.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
            setUploadError(`File too large (Max: ${MAX_FILE_SIZE_MB} MB)`);
            setSelectedFile(null);
            setUploadedFileName('');
            setUploadedFileId(null);
            if(fileInputRef.current) fileInputRef.current.value = ''; // Clear input
            return;
        }

        // If validation passes:
        setSelectedFile(file);
        setUploadedFileName(file.name); // Show the name immediately
        handleFileUpload(file); // Automatically start upload

        // Clear the file input value so the same file can be selected again if needed
        if(fileInputRef.current) fileInputRef.current.value = '';
    };

    const handleFileUpload = async (fileToUpload) => {
        if (!fileToUpload || !isAuthenticated) return; // Need auth to upload

        setIsUploading(true);
        setUploadError(null);
        setUploadedFileId(null); // Clear previous ID during new upload

        try {
            // Call the API function (ensure it's defined in ragApi.js)
            // This function should handle FormData creation and the fetch call
            const result = await uploadTemporaryFile(fileToUpload);

            if (result && result.file_id) {
                setUploadedFileId(result.file_id); // Store the unique ID from the backend
                setUploadedFileName(result.filename); // Confirm with filename from backend response
                console.log('Temporary file uploaded successfully. ID:', result.file_id);
            } else {
                throw new Error("Upload response did not contain a file ID.");
            }
        } catch (err) {
            console.error("Temporary file upload error:", err);
            setUploadError(err.message || 'An error occurred during upload.');
            // Clear file state on error
            setSelectedFile(null);
            setUploadedFileName('');
            setUploadedFileId(null);
        } finally {
            setIsUploading(false);
        }
    };

    // Trigger the hidden file input
    const triggerFileInput = () => {
        if (!isUploading) { // Don't trigger if already uploading
           fileInputRef.current?.click();
        }
    };

    // Cancel/remove the uploaded file before sending
    const cancelFileUpload = () => {
        setSelectedFile(null);
        setUploadedFileId(null);
        setUploadedFileName('');
        setUploadError(null);
        setIsUploading(false); // Ensure uploading state is reset
        // No need to inform backend, file expires automatically
    };
    // --------------------------


    // --- Main Send Handler (Unchanged) ---
    const handleSend = useCallback(async (queryOverride = '') => {
        // Use queryOverride if provided (e.g., from suggested actions), otherwise use inputValue
        const queryToSend = (queryOverride || inputValue).trim();
        const fileIdToSend = uploadedFileId; // Capture the current file ID

        // Don't send if loading, not authenticated, or if there's no text AND no file attached
        if (isLoading || !isAuthenticated || (!queryToSend && !fileIdToSend)) {
            console.log("Send condition not met:", {isLoading, isAuthenticated, queryToSend, fileIdToSend});
            return;
        }

        // --- Prepare message and state updates ---
        const userMessageText = queryToSend || `(Query based on file: ${uploadedFileName})`; // Use filename if text is empty but file exists
        const userMessage = { sender: 'user', text: userMessageText, timestamp: new Date().toISOString() };
        let conversationIdToUse = activeConversationId;
        let isNewChatFlow = false;
        const previousMessages = messages; // Store previous messages for potential rollback

        // Update UI immediately
        setMessages(prevMessages => [...prevMessages, userMessage]);
        setInputValue(''); // Clear input field
        setSuggestedActions([]); // Clear suggestions
        setIsLoading(true); // Set loading state

        // --- Clear file state AFTER capturing fileIdToSend and updating UI ---
        setSelectedFile(null);
        setUploadedFileId(null);
        setUploadedFileName('');
        setUploadError(null);
        // --------------------------------------------------------------------

        try {
            // --- Handle creating a new conversation if needed ---
            if (!activeConversationId) {
                isNewChatFlow = true;
                console.log("Creating new conversation with first message:", userMessage);
                // Prepare data for saving the *first* message (user message)
                const newChatData = { conversation_id: null, title: null, messages: [userMessage] };
                const saveResponse = await saveConversation(newChatData); // Save to get ID and Title

                if (saveResponse && saveResponse.conversation_id) {
                    conversationIdToUse = saveResponse.conversation_id;
                    const newTitle = saveResponse.title || `Chat ${new Date().toLocaleTimeString()}`;
                    setActiveConversationId(conversationIdToUse);
                    setCurrentTitle(newTitle);
                    // Add new conversation to the list and sort
                    setConversationList(prevList =>
                        [{ id: conversationIdToUse, title: newTitle, timestamp: new Date().toISOString() }, ...prevList]
                        .sort((a, b) => (new Date(b.timestamp || 0)) - (new Date(a.timestamp || 0)))
                    );
                    console.log("New conversation created:", conversationIdToUse, newTitle);
                } else {
                    // If backend fails to return ID, throw error to trigger rollback
                    throw new Error("Backend did not return an ID for the new chat.");
                }
            }

            // --- Fetch the AI response using the conversation ID and potentially the file ID ---
            console.log(`Fetching query response for convo: ${conversationIdToUse}, File ID: ${fileIdToSend}`);
            // *** Ensure fetchQueryResponse in ragApi.js accepts fileIdToSend ***
            const response = await fetchQueryResponse(queryToSend, conversationIdToUse, fileIdToSend);

            const aiMessage = {
                sender: 'ai',
                text: response.answer || "Sorry, I couldn't get a response.",
                sources: response.sources || [],
                timestamp: new Date().toISOString()
            };
            setMessages(prevMessages => [...prevMessages, aiMessage]); // Add AI message
            setSuggestedActions(response.suggestions || []); // Update suggestions

        } catch (error) {
            console.error("Error during send/response:", error);
             // --- Rollback UI state if new chat creation failed ---
             if (isNewChatFlow && !conversationIdToUse) {
                 console.warn("Reverting state due to new chat creation failure.");
                 setMessages(previousMessages); // Restore previous messages
                 setActiveConversationId(null); // No active conversation
                 setCurrentTitle("Chat with Symphony"); // Reset title
             }
             // --- Display error message ---
            const errorMsg = { sender: 'ai', text: `⚠️ Error: ${error.message || 'Could not reach server.'}`, error: true, timestamp: new Date().toISOString() };
            setMessages(prevMessages => [...prevMessages, errorMsg]);
            setSuggestedActions([]); // Clear suggestions on error
        } finally {
            setIsLoading(false); // Stop loading indicator
        }
    // Include file upload state variables in dependencies if they affect the logic directly
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [inputValue, isLoading, isAuthenticated, activeConversationId, messages, uploadedFileId, uploadedFileName]); // Added file state


    const handleSelectConversation = useCallback(async (conversationId) => {
        if (isLoading || conversationId === activeConversationId || !isAuthenticated) return;

        // Clear any pending save timeout
        if (saveTimeoutRef.current) { clearTimeout(saveTimeoutRef.current); saveTimeoutRef.current = null; }
        // Clear any pending file upload state
        cancelFileUpload();

        console.log(`Selecting conversation: ${conversationId}`);
        setIsLoading(true);
        setMessages([]); // Clear current messages
        setSuggestedActions([]); // Clear suggestions
        setCurrentTitle("Loading..."); // Set loading title
        setActiveConversationId(conversationId); // Set the new active ID

        try {
            const conversationData = await fetchConversation(conversationId);
            if (conversationData && conversationData.id === conversationId) { // Verify ID match
                console.log(`Loaded ${conversationData.messages?.length || 0} messages for ${conversationId}`);
                setMessages(conversationData.messages || []); // Load messages
                setCurrentTitle(conversationData.title || "Chat"); // Load title

                // Update timestamp in the list to bring selected item to top after sort
                setConversationList(prevList => prevList.map(conv =>
                    conv.id === conversationId ? { ...conv, timestamp: new Date().toISOString() } : conv
                ).sort((a, b) => (new Date(b.timestamp || 0)) - (new Date(a.timestamp || 0))));

            } else {
                throw new Error(`Conversation data missing or ID mismatch for ID: ${conversationId}`);
            }
        } catch (error) {
            console.error("Error fetching conversation:", error);
            setCurrentTitle("Error Loading Chat");
            setActiveConversationId(null); // Reset active ID on error
            setMessages([{ sender: 'ai', text: `⚠️ Error loading conversation. ${error.message}`, error: true, timestamp: new Date().toISOString() }]);
        } finally {
            setIsLoading(false); // Stop loading indicator
        }
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [activeConversationId, isLoading, isAuthenticated]); // Dependencies


    const handleNewChat = useCallback(() => {
        if (isLoading) return; // Don't allow new chat while loading
        // Clear pending save
        if (saveTimeoutRef.current) { clearTimeout(saveTimeoutRef.current); saveTimeoutRef.current = null; }
        // Clear file upload state
        cancelFileUpload();

        console.log("Resetting UI for New Chat");
        setMessages([]);
        setInputValue('');
        setIsLoading(false); // Ensure loading is off
        setIsSaving(false); // Ensure saving is off
        setActiveConversationId(null); // No active conversation
        setCurrentTitle("Chat with Symphony"); // Reset title
        setSuggestedActions([]); // Clear suggestions
        // Optionally focus the textarea
        textareaRef.current?.focus();
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [isLoading]); // Dependency


    // --- Effects (Unchanged) ---
    // Initial load effect
    useEffect(() => {
        const storedToken = localStorage.getItem('authToken');
        if (storedToken) {
            setToken(storedToken);
            loadInitialConversations(storedToken); // Attempt to load data if token exists
        } else {
             setIsAuthLoading(false); // No token, stop auth loading
             setIsAuthenticated(false); // Ensure not authenticated
        }
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []); // Run only once on mount

    // Clear suggestions when input changes while not loading
    useEffect(() => {
        if (inputValue && !isLoading) {
            setSuggestedActions([]);
        }
    }, [inputValue, isLoading]);

    // Debounced save effect
     useEffect(() => {
         // Trigger save only when authenticated, an active chat exists, and messages are present
         if (isAuthenticated && activeConversationId && messages.length > 0) {
             triggerSaveConversation();
         }
         // Cleanup function to clear timeout if component unmounts or dependencies change
         return () => {
             if (saveTimeoutRef.current) {
                 clearTimeout(saveTimeoutRef.current);
             }
         };
     }, [messages, activeConversationId, triggerSaveConversation, isAuthenticated]); // Rerun when these change


    // --- Input Area KeyDown Handler ---
    const handleInputKeyDown = (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault(); // Prevent newline
            // Check if send is possible before triggering
            const queryToSend = inputValue.trim();
            const fileIdToSend = uploadedFileId;
            if (!isLoading && isAuthenticated && (queryToSend || fileIdToSend)) {
                handleSend(); // Trigger send
            }
        }
    };

    // --- Render Logic ---
    const showSuggestedActions = !isLoading && isAuthenticated && messages.length > 0 && suggestedActions.length > 0;

    // Loading screen before authentication check completes
    if (isAuthLoading && !token) { // Show only if loading AND no token yet
         return (
             <div className="loading-fullscreen">
                 <img src="/Brand mark - transparent - png.png" alt="Symphony Logo" className="auth-logo" />
                 <p>Loading...</p>
             </div>
         );
    }

    return (
        <>
            {!isAuthenticated ? (
                // Render AuthForm if not authenticated
                <AuthForm
                    isLoginMode={showLogin}
                    onSubmit={showLogin ? handleLogin : handleRegister}
                    error={authError}
                    onToggleMode={toggleAuthMode}
                    isLoading={isAuthLoading} // Use auth loading state
                 />
            ) : (
                // Render main chat interface if authenticated
                <div className="app-container">
                    <Sidebar
                        conversationList={conversationList}
                        activeConversationId={activeConversationId}
                        onSelectConversation={handleSelectConversation}
                        onNewChat={handleNewChat}
                        onLogout={handleLogout}
                        isAuthenticated={isAuthenticated}
                    />
                    <div className="chat-area">
                        <ChatHeader title={currentTitle} />
                        <MessageList messages={messages} />
                        {/* Loading indicator */}
                        {isLoading && <div className="loading-indicator">Symphony is thinking...</div>}
                        {/* Suggested Actions */}
                        {showSuggestedActions && <SuggestedActions actions={suggestedActions} onActionClick={handleSuggestedAction} isLoading={isLoading} />}

                        {/* --- Input Area Wrapper --- */}
                        <div className="input-area-wrapper">
                            {/* File Upload Status/Error */}
                             {uploadError && <p className="upload-error">Upload Error: {uploadError}</p>}
                             {uploadedFileName && !uploadError && (
                                <div className="upload-status">
                                    Attached: {uploadedFileName} ({uploadedFileId && !isUploading ? 'Ready' : 'Uploading...'})
                                    {/* Show cancel button only if upload is complete or selected */}
                                    {!isUploading && (
                                        <button onClick={cancelFileUpload} className="cancel-upload-button" title="Remove File">
                                            <FaTimesCircle />
                                        </button>
                                    )}
                                 </div>
                             )}

                            {/* Input controls container */}
                            <div className="input-area">
                                {/* Hidden File Input */}
                                <input
                                  type="file"
                                  ref={fileInputRef}
                                  onChange={handleFileChange}
                                  style={{ display: 'none' }}
                                  accept={ALLOWED_MIME_TYPES.join(',')}
                                  disabled={isUploading || isLoading}
                                />
                                {/* Text Input Area Component */}
                                <InputArea
                                    inputValue={inputValue}
                                    onInputChange={handleInputChange}
                                    isLoading={isLoading || isUploading} // Disable input while uploading too
                                    textareaRef={textareaRef}
                                    onKeyDown={handleInputKeyDown} // Pass keydown handler
                                />
                                {/* Upload Button */}
                                <button
                                  onClick={triggerFileInput}
                                  disabled={isUploading || isLoading}
                                  className="upload-button" // Use specific class for styling
                                  title={`Attach File (${ALLOWED_MIME_TYPES.map(t => t.split('/')[1]).join(', ')})`}
                                  aria-label="Attach file"
                                >
                                  {isUploading ? '...' : <FaPaperclip />}
                                </button>
                                {/* Send Button */}
                                <button
                                    className="send-button" // Use specific class for styling
                                    onClick={() => handleSend()} // Trigger send directly
                                    disabled={isLoading || isUploading || (!inputValue.trim() && !uploadedFileId)} // Disable if no text AND no file
                                    aria-label="Send message"
                                >
                                    {/* Add text "Send" along with the icon */}
                                    <span>Send</span>
                                    <FaPaperPlane style={{ marginLeft: '5px' }}/>
                                </button>
                            </div>
                        </div>
                        {/* --- End Input Area Wrapper --- */}
                    </div>
                </div>
            )}
        </>
    );
}

export default App;
