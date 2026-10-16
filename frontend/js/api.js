/**
 * API client module for AI Cybersecurity Assistant.
 * Handles same-origin requests to /api/v1/health and /api/v1/chat with robust error translation.
 */

(function (window) {
    'use strict';

    const API_BASE = '/api/v1';

    /**
     * Custom error class representing API failures with user-friendly descriptions.
     */
    class AssistantAPIError extends Error {
        constructor(message, statusCode = null, details = null) {
            super(message);
            this.name = 'AssistantAPIError';
            this.statusCode = statusCode;
            this.details = details;
        }
    }

    /**
     * Fetch backend health status without billable API calls.
     * @returns {Promise<Object>} HealthResponse payload.
     */
    async function getHealth() {
        try {
            const response = await fetch(`${API_BASE}/health`, {
                method: 'GET',
                headers: {
                    'Accept': 'application/json'
                }
            });

            // Endpoint returns 200 (healthy) or 503 (degraded) with complete diagnostic JSON
            const data = await response.json();
            return {
                ok: response.ok,
                status: response.status,
                data: data
            };
        } catch (err) {
            return {
                ok: false,
                status: null,
                error: 'Cannot connect to AI Assistant backend at 127.0.0.1:8000. Please ensure Uvicorn is running.',
                data: null
            };
        }
    }

    /**
     * Send a cybersecurity question to the RAG chat endpoint.
     * @param {string} message - User question text (1-2000 characters).
     * @param {string|null} conversationId - Optional session ID for multi-turn context.
     * @returns {Promise<Object>} ChatAPIResponse payload.
     */
    async function sendChatMessage(message, conversationId = null) {
        if (!message || typeof message !== 'string' || !message.trim()) {
            throw new AssistantAPIError('Query string cannot be empty or whitespace-only.', 422);
        }

        const payload = {
            message: message.trim()
        };

        if (conversationId && typeof conversationId === 'string' && conversationId.trim()) {
            payload.conversation_id = conversationId.trim();
        }

        let response;
        try {
            response = await fetch(`${API_BASE}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify(payload)
            });
        } catch (networkErr) {
            throw new AssistantAPIError(
                'Network connection failure: Unable to reach the backend at 127.0.0.1:8000.',
                null,
                networkErr.message
            );
        }

        let responseData;
        try {
            responseData = await response.json();
        } catch (jsonErr) {
            throw new AssistantAPIError(
                `Server returned status ${response.status} with non-JSON response.`,
                response.status
            );
        }

        if (!response.ok) {
            // HTTP 422 Validation Error
            if (response.status === 422) {
                let errorMsg = 'Validation error: please check your query input.';
                if (responseData && Array.isArray(responseData.errors) && responseData.errors.length > 0) {
                    errorMsg = responseData.errors.join('; ');
                } else if (responseData && responseData.detail) {
                    errorMsg = typeof responseData.detail === 'string' 
                        ? responseData.detail 
                        : JSON.stringify(responseData.detail);
                }
                throw new AssistantAPIError(errorMsg, 422, responseData);
            }

            // HTTP 503 Provider / Transient Error
            if (response.status === 503) {
                const errorMsg = (responseData && responseData.detail) 
                    ? responseData.detail 
                    : 'The LLM service is temporarily unavailable. Please retry shortly.';
                throw new AssistantAPIError(errorMsg, 503, responseData);
            }

            // HTTP 500 / Unexpected Internal Error
            if (response.status === 500) {
                const errorMsg = (responseData && responseData.detail) 
                    ? responseData.detail 
                    : 'An unexpected internal error occurred while processing your query.';
                throw new AssistantAPIError(errorMsg, 500, responseData);
            }

            // Other HTTP Errors (400, etc.)
            const fallbackMsg = (responseData && responseData.detail) 
                ? responseData.detail 
                : `Request failed with HTTP status ${response.status}.`;
            throw new AssistantAPIError(fallbackMsg, response.status, responseData);
        }

        return responseData;
    }

    // Expose API module globally
    window.AssistantAPI = {
        getHealth,
        sendChatMessage,
        AssistantAPIError
    };

})(window);
