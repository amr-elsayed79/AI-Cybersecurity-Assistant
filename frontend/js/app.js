/**
 * UI Controller and DOM State Management for AI Cybersecurity Assistant Dashboard.
 * 
 * SECURITY GUARANTEES:
 * 1. Safe DOM construction only (textContent, createElement). Never uses innerHTML for user/API strings.
 * 2. Session ID is kept strictly in JavaScript runtime memory (never in localStorage/sessionStorage/cookies).
 * 3. Clearly labels Mock Provider mode to prevent misinterpreting test workflows as verified intelligence.
 */

(function () {
    'use strict';

    // Runtime State (Ephemeral, in-memory only)
    let activeConversationId = null;
    let isRequestPending = false;
    let configuredProvider = 'unknown';

    // DOM Elements
    const elements = {
        // Header & Status
        llmModeBadge: document.getElementById('llm-mode-badge'),
        llmModeText: document.getElementById('llm-mode-text'),
        systemHealthPill: document.getElementById('system-health-pill'),
        systemHealthText: document.getElementById('system-health-text'),
        mockWarningBanner: document.getElementById('mock-warning-banner'),
        
        // Chat Area
        activeSessionId: document.getElementById('active-session-id'),
        newChatBtn: document.getElementById('new-chat-btn'),
        chatMessages: document.getElementById('chat-messages'),
        welcomeMessage: document.getElementById('welcome-message'),
        chatStatusBar: document.getElementById('chat-status-bar'),
        loadingSpinner: document.getElementById('loading-spinner'),
        errorAlert: document.getElementById('error-alert'),
        errorMessage: document.getElementById('error-message'),
        chatForm: document.getElementById('chat-form'),
        chatInput: document.getElementById('chat-input'),
        charCounter: document.getElementById('char-counter'),
        sendBtn: document.getElementById('send-btn'),
        quickPromptChips: document.querySelectorAll('.quick-prompt-chip'),

        // Evidence & Diagnostics
        tabBtnEvidence: document.getElementById('tab-btn-evidence'),
        tabBtnDiagnostics: document.getElementById('tab-btn-diagnostics'),
        tabContentEvidence: document.getElementById('tab-content-evidence'),
        tabContentDiagnostics: document.getElementById('tab-content-diagnostics'),
        turnStatusBadge: document.getElementById('turn-status-badge'),
        turnMetricsRow: document.getElementById('turn-metrics-row'),
        turnLang: document.getElementById('turn-lang'),
        turnRetrievalTime: document.getElementById('turn-retrieval-time'),
        turnLlmTime: document.getElementById('turn-llm-time'),
        turnTotalTime: document.getElementById('turn-total-time'),
        turnStatusDesc: document.getElementById('turn-status-desc'),
        citedCountBadge: document.getElementById('cited-count-badge'),
        citedSourcesList: document.getElementById('cited-sources-list'),
        retrievedCountBadge: document.getElementById('retrieved-count-badge'),
        retrievedSourcesList: document.getElementById('retrieved-sources-list'),
        
        // Diagnostics Panel
        refreshHealthBtn: document.getElementById('refresh-health-btn'),
        diagStatus: document.getElementById('diag-status'),
        diagVersion: document.getElementById('diag-version'),
        diagKbVerified: document.getElementById('diag-kb-verified'),
        diagVectorStatus: document.getElementById('diag-vector-status'),
        diagPointsCount: document.getElementById('diag-points-count'),
        diagLlmProvider: document.getElementById('diag-llm-provider'),
        diagLlmConfigured: document.getElementById('diag-llm-configured')
    };

    /**
     * Helper to detect if a text string contains Arabic Unicode characters.
     */
    function isArabicText(text) {
        if (!text) return false;
        // Arabic Unicode block: \u0600-\u06FF, \u0750-\u077F, \u08A0-\u08FF, \uFB50-\uFDFF, \uFE70-\uFEFF
        const arabicRegex = /[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]/;
        return arabicRegex.test(text);
    }

    /**
     * Safely format message content by splitting paragraphs and preserving formatting without innerHTML.
     */
    function appendSafeContentToBubble(bubbleElement, rawText, isArabic) {
        if (!rawText) return;

        const paragraphs = rawText.split(/\n\n+/);
        paragraphs.forEach(para => {
            const p = document.createElement('p');
            p.style.marginBottom = '8px';

            const lines = para.split('\n');
            lines.forEach((line, idx) => {
                if (idx > 0) {
                    p.appendChild(document.createElement('br'));
                }
                const textNode = document.createTextNode(line);
                p.appendChild(textNode);
            });

            bubbleElement.appendChild(p);
        });
    }

    /**
     * Render a message turn in the chat feed safely using DOM APIs.
     */
    function renderMessage(role, content, metadata = null) {
        // Hide initial welcome card once messages start flowing
        if (elements.welcomeMessage) {
            elements.welcomeMessage.style.display = 'none';
        }

        const isUser = (role === 'user');
        const isArabic = isArabicText(content);

        const row = document.createElement('div');
        row.className = `message-row ${isUser ? 'user' : 'assistant'}`;

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = isUser ? 'U' : 'AI';
        row.appendChild(avatar);

        const bubble = document.createElement('div');
        bubble.className = `message-bubble ${isArabic ? 'bubble-ar' : 'bubble-en'}`;

        // Site-specific disclaimer callout if flagged
        if (!isUser && metadata && metadata.is_site_specific) {
            const disclaimerBox = document.createElement('div');
            disclaimerBox.className = 'site-disclaimer-box';
            disclaimerBox.textContent = isArabic
                ? 'تنبيه: قاعدة المعرفة الأمنية تقدم إرشادات عامة ولا يمكنها تأكيد وجود ثغرة في موقع أو خادم محدد.'
                : 'Notice: Knowledge Base information is defensive guidance and cannot confirm vulnerabilities on live targets.';
            bubble.appendChild(disclaimerBox);
        }

        // Message text body
        appendSafeContentToBubble(bubble, content, isArabic);

        // Assistant Turn Footer with Grounding Badge & Latency
        if (!isUser && metadata) {
            const footer = document.createElement('div');
            footer.className = 'bubble-meta-footer';

            const statusBadge = document.createElement('span');
            statusBadge.className = `badge ${getStatusBadgeClass(metadata.grounding_status)}`;
            statusBadge.textContent = formatGroundingStatus(metadata.grounding_status);
            footer.appendChild(statusBadge);

            if (metadata.total_latency_ms !== undefined) {
                const latencySpan = document.createElement('span');
                latencySpan.textContent = `• ${metadata.total_latency_ms} ms`;
                footer.appendChild(latencySpan);
            }

            bubble.appendChild(footer);
        }

        row.appendChild(bubble);
        elements.chatMessages.appendChild(row);

        // Scroll feed to bottom
        elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
    }

    /**
     * Map grounding status enum to CSS class.
     */
    function getStatusBadgeClass(status) {
        switch (status) {
            case 'grounded':
                return 'badge-grounded';
            case 'site_qualified':
                return 'badge-site-qualified';
            case 'insufficient_evidence':
            case 'empty_retrieval':
                return 'badge-refusal';
            case 'attribution_failed':
            case 'incomplete_truncated':
                return 'badge-attribution-failed';
            default:
                return 'badge-neutral';
        }
    }

    /**
     * Map grounding status enum to human-readable label.
     */
    function formatGroundingStatus(status) {
        switch (status) {
            case 'grounded':
                return 'Grounded (Verified KB)';
            case 'site_qualified':
                return 'Site-Specific Qualified';
            case 'insufficient_evidence':
                return 'Insufficient KB Evidence';
            case 'empty_retrieval':
                return 'No Relevant Evidence Found';
            case 'attribution_failed':
                return 'Attribution Check Suppressed';
            case 'incomplete_truncated':
                return 'Incomplete / Token Truncated';
            default:
                return status || 'Completed';
        }
    }

    /**
     * Safely render individual source cards in the Evidence drawer.
     */
    function renderSourceCard(source) {
        const card = document.createElement('div');
        card.className = 'source-card';

        // Header: Name and Retrieval Score
        const header = document.createElement('div');
        header.className = 'source-card-header';

        const title = document.createElement('div');
        title.className = 'source-title';
        title.textContent = source.name || source.vulnerability_id || 'Knowledge Record';
        header.appendChild(title);

        if (source.score !== undefined && source.score !== null) {
            const scorePill = document.createElement('div');
            scorePill.className = 'source-score-pill';
            scorePill.textContent = `Score: ${Number(source.score).toFixed(4)}`;
            header.appendChild(scorePill);
        }
        card.appendChild(header);

        // Metadata Tags: Section, CWE, OWASP
        const metaRow = document.createElement('div');
        metaRow.className = 'source-meta-row';

        if (source.section) {
            const sectionTag = document.createElement('span');
            sectionTag.className = 'source-tag';
            sectionTag.textContent = `Section: ${source.section}`;
            metaRow.appendChild(sectionTag);
        }

        if (Array.isArray(source.cwe) && source.cwe.length > 0) {
            source.cwe.forEach(cweId => {
                const cweTag = document.createElement('span');
                cweTag.className = 'source-tag source-tag-cwe';
                cweTag.textContent = cweId;
                metaRow.appendChild(cweTag);
            });
        }

        if (Array.isArray(source.owasp) && source.owasp.length > 0) {
            source.owasp.forEach(owaspId => {
                const owaspTag = document.createElement('span');
                owaspTag.className = 'source-tag source-tag-owasp';
                owaspTag.textContent = owaspId;
                metaRow.appendChild(owaspTag);
            });
        }
        card.appendChild(metaRow);

        // Chunk ID footer
        if (source.chunk_id) {
            const chunkIdElem = document.createElement('div');
            chunkIdElem.className = 'source-chunk-id';
            chunkIdElem.textContent = `Chunk: ${source.chunk_id}`;
            card.appendChild(chunkIdElem);
        }

        return card;
    }

    /**
     * Update the Evidence panel for the active turn using real response payloads.
     */
    function updateEvidencePanel(response) {
        // Update Attribution Status Badge
        elements.turnStatusBadge.className = `badge ${getStatusBadgeClass(response.grounding_status)}`;
        elements.turnStatusBadge.textContent = formatGroundingStatus(response.grounding_status);

        // Update Latency & Language Metrics
        elements.turnMetricsRow.classList.remove('hidden');
        elements.turnLang.textContent = (response.language === 'ar') ? 'Arabic' : 'English';
        elements.turnRetrievalTime.textContent = `${response.retrieval_latency_ms || 0} ms`;
        elements.turnLlmTime.textContent = `${response.llm_latency_ms || 0} ms`;
        elements.turnTotalTime.textContent = `${response.total_latency_ms || 0} ms`;

        // Update Description text based on operational state
        if (response.grounding_status === 'grounded') {
            elements.turnStatusDesc.textContent = 'All citations verified against retrieved Master KB chunks. Guardrails passed.';
        } else if (response.grounding_status === 'site_qualified') {
            elements.turnStatusDesc.textContent = 'Site-specific target inquiry detected. Answer qualified with mandatory assessment disclaimer.';
        } else if (response.grounding_status === 'empty_retrieval' || response.grounding_status === 'insufficient_evidence') {
            elements.turnStatusDesc.textContent = 'Retrieval context was insufficient or below relevance threshold. Refusal returned safely.';
        } else if (response.grounding_status === 'attribution_failed') {
            elements.turnStatusDesc.textContent = 'Generated narrative cited unverified tags or unsupported identifiers. Answer suppressed for safety.';
        } else if (response.grounding_status === 'incomplete_truncated') {
            elements.turnStatusDesc.textContent = 'Generation reached provider token limit and was cut off. Partial answer was suppressed for safety.';
        }

        // Render Cited Sources
        elements.citedSourcesList.textContent = '';
        const citedSources = response.cited_sources || [];
        elements.citedCountBadge.textContent = citedSources.length;

        if (citedSources.length === 0) {
            const emptyDiv = document.createElement('div');
            emptyDiv.className = 'sources-empty';
            emptyDiv.textContent = 'No sources explicitly cited in this answer.';
            elements.citedSourcesList.appendChild(emptyDiv);
        } else {
            citedSources.forEach(src => {
                elements.citedSourcesList.appendChild(renderSourceCard(src));
            });
        }

        // Render Retrieved Candidates
        elements.retrievedSourcesList.textContent = '';
        const retrievedSources = response.retrieved_sources || [];
        elements.retrievedCountBadge.textContent = retrievedSources.length;

        if (retrievedSources.length === 0) {
            const emptyDiv = document.createElement('div');
            emptyDiv.className = 'sources-empty';
            emptyDiv.textContent = 'No candidate chunks retrieved.';
            elements.retrievedSourcesList.appendChild(emptyDiv);
        } else {
            retrievedSources.forEach(src => {
                elements.retrievedSourcesList.appendChild(renderSourceCard(src));
            });
        }
    }

    /**
     * Clear active conversation state and reset UI (New Chat action).
     */
    function resetConversation() {
        activeConversationId = null;
        elements.activeSessionId.textContent = 'Fresh Conversation';
        elements.activeSessionId.title = 'No active conversation ID yet';

        // Reset chat feed
        elements.chatMessages.textContent = '';
        if (elements.welcomeMessage) {
            elements.welcomeMessage.style.display = 'block';
            elements.chatMessages.appendChild(elements.welcomeMessage);
        }

        // Reset evidence panel
        elements.turnStatusBadge.className = 'badge badge-neutral';
        elements.turnStatusBadge.textContent = 'No Active Query';
        elements.turnMetricsRow.classList.add('hidden');
        elements.turnStatusDesc.textContent = 'Submit a question to inspect retrieved Knowledge Base records, citations, and attribution verification.';
        
        elements.citedCountBadge.textContent = '0';
        elements.citedSourcesList.textContent = '';
        const emptyCited = document.createElement('div');
        emptyCited.className = 'sources-empty';
        emptyCited.textContent = 'No cited sources for this turn.';
        elements.citedSourcesList.appendChild(emptyCited);

        elements.retrievedCountBadge.textContent = '0';
        elements.retrievedSourcesList.textContent = '';
        const emptyRet = document.createElement('div');
        emptyRet.className = 'sources-empty';
        emptyRet.textContent = 'No retrieved candidates.';
        elements.retrievedSourcesList.appendChild(emptyRet);

        // Hide any active error or status
        elements.chatStatusBar.classList.add('hidden');
        elements.errorAlert.classList.add('hidden');
        elements.chatInput.value = '';
        elements.charCounter.textContent = '0 / 2000';
        elements.chatInput.focus();
    }

    /**
     * Check backend health and update status badges / diagnostics.
     */
    async function checkHealthStatus() {
        elements.systemHealthPill.className = 'health-pill health-unknown';
        elements.systemHealthText.textContent = 'Checking...';

        const result = await window.AssistantAPI.getHealth();

        if (!result.ok && !result.data) {
            // Complete network / connection failure
            elements.systemHealthPill.className = 'health-pill health-offline';
            elements.systemHealthText.textContent = 'Offline (127.0.0.1:8000)';
            elements.llmModeBadge.className = 'mode-badge mode-unknown';
            elements.llmModeText.textContent = 'Unknown Provider';
            elements.mockWarningBanner.classList.add('hidden');

            elements.diagStatus.textContent = 'Unreachable';
            elements.diagStatus.style.color = 'var(--color-danger)';
            return;
        }

        const data = result.data;
        configuredProvider = (data.llm_provider || 'unknown').toLowerCase();

        // 1. Update Health Indicator
        if (data.status === 'healthy') {
            elements.systemHealthPill.className = 'health-pill health-healthy';
            elements.systemHealthText.textContent = 'Healthy (588 KB Chunks)';
            elements.diagStatus.textContent = 'Healthy (200 OK)';
            elements.diagStatus.style.color = 'var(--color-success)';
        } else {
            elements.systemHealthPill.className = 'health-pill health-degraded';
            elements.systemHealthText.textContent = 'Degraded';
            elements.diagStatus.textContent = 'Degraded (503 Service Warning)';
            elements.diagStatus.style.color = 'var(--color-warning)';
        }

        // 2. Update Provider Indicator & Prominent Mock Warning
        if (configuredProvider === 'mock') {
            elements.llmModeBadge.className = 'mode-badge mode-mock';
            elements.llmModeText.textContent = 'Provider: Mock Simulator';
            elements.mockWarningBanner.classList.remove('hidden');
        } else if (configuredProvider === 'groq') {
            elements.llmModeBadge.className = 'mode-badge mode-groq';
            elements.llmModeText.textContent = 'Provider: Groq';
            elements.mockWarningBanner.classList.add('hidden');
        } else {
            elements.llmModeBadge.className = 'mode-badge mode-unknown';
            elements.llmModeText.textContent = `Provider: ${configuredProvider}`;
            elements.mockWarningBanner.classList.add('hidden');
        }

        // 3. Update Diagnostics Details (never disclosing secrets)
        elements.diagVersion.textContent = data.version || '0.2.0';
        elements.diagKbVerified.textContent = data.master_kb_checksum_verified ? 'Verified Baseline Match' : 'Checksum Mismatch';
        elements.diagVectorStatus.textContent = data.vector_store_status || 'Unknown';
        elements.diagPointsCount.textContent = `${data.indexed_point_count || 0} Points`;
        elements.diagLlmProvider.textContent = data.llm_provider || 'None';
        elements.diagLlmConfigured.textContent = data.llm_configured ? 'Configured (Active)' : 'Unconfigured';
    }

    /**
     * Submit user question to RAG chat pipeline.
     */
    async function handleChatSubmit() {
        if (isRequestPending) return;

        const message = elements.chatInput.value.trim();
        if (!message) return;

        // Set pending UI state
        isRequestPending = true;
        elements.sendBtn.disabled = true;
        elements.chatInput.disabled = true;
        elements.chatStatusBar.classList.remove('hidden');
        elements.loadingSpinner.classList.remove('hidden');
        elements.errorAlert.classList.add('hidden');

        // Render user message immediately
        renderMessage('user', message);
        elements.chatInput.value = '';
        elements.charCounter.textContent = '0 / 2000';

        try {
            const response = await window.AssistantAPI.sendChatMessage(message, activeConversationId);

            // Update in-memory session identifier
            if (response.conversation_id) {
                activeConversationId = response.conversation_id;
                elements.activeSessionId.textContent = activeConversationId;
                elements.activeSessionId.title = `Active in-memory session: ${activeConversationId}`;
            }

            // Render assistant narrative
            renderMessage('assistant', response.answer, {
                grounding_status: response.grounding_status,
                is_site_specific: response.is_site_specific,
                total_latency_ms: response.total_latency_ms,
                language: response.language
            });

            // Update Evidence panel
            updateEvidencePanel(response);

        } catch (err) {
            // Show user-friendly error without stack traces or path disclosure
            elements.errorAlert.classList.remove('hidden');
            elements.errorMessage.textContent = err.message || 'An unexpected error occurred while communicating with the assistant.';
        } finally {
            // Restore input UI state
            isRequestPending = false;
            elements.sendBtn.disabled = false;
            elements.chatInput.disabled = false;
            elements.loadingSpinner.classList.add('hidden');
            if (elements.errorAlert.classList.contains('hidden')) {
                elements.chatStatusBar.classList.add('hidden');
            }
            elements.chatInput.focus();
        }
    }

    /**
     * Tab switching logic for Evidence vs Diagnostics.
     */
    function setupTabs() {
        elements.tabBtnEvidence.addEventListener('click', () => {
            elements.tabBtnEvidence.classList.add('active');
            elements.tabBtnEvidence.setAttribute('aria-selected', 'true');
            elements.tabBtnDiagnostics.classList.remove('active');
            elements.tabBtnDiagnostics.setAttribute('aria-selected', 'false');

            elements.tabContentEvidence.classList.add('active');
            elements.tabContentDiagnostics.classList.remove('active');
        });

        elements.tabBtnDiagnostics.addEventListener('click', () => {
            elements.tabBtnDiagnostics.classList.add('active');
            elements.tabBtnDiagnostics.setAttribute('aria-selected', 'true');
            elements.tabBtnEvidence.classList.remove('active');
            elements.tabBtnEvidence.setAttribute('aria-selected', 'false');

            elements.tabContentDiagnostics.classList.add('active');
            elements.tabContentEvidence.classList.remove('active');
        });
    }

    /**
     * Setup event listeners for forms, buttons, and keyboard controls.
     */
    function setupEventListeners() {
        // Chat Form Submit
        elements.chatForm.addEventListener('submit', (e) => {
            e.preventDefault();
            handleChatSubmit();
        });

        // Keydown handler: Enter sends, Shift+Enter adds newline
        elements.chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleChatSubmit();
            }
        });

        // Input character counter
        elements.chatInput.addEventListener('input', () => {
            const length = elements.chatInput.value.length;
            elements.charCounter.textContent = `${length} / 2000`;
        });

        // New Chat Button
        elements.newChatBtn.addEventListener('click', () => {
            resetConversation();
        });

        // Refresh Health Button
        elements.refreshHealthBtn.addEventListener('click', () => {
            checkHealthStatus();
        });

        // Quick Prompt Chips
        elements.quickPromptChips.forEach(chip => {
            chip.addEventListener('click', () => {
                const promptText = chip.getAttribute('data-prompt');
                if (promptText && !isRequestPending) {
                    elements.chatInput.value = promptText;
                    elements.charCounter.textContent = `${promptText.length} / 2000`;
                    handleChatSubmit();
                }
            });
        });
    }

    /**
     * Initialize application on DOM content loaded.
     */
    function init() {
        setupTabs();
        setupEventListeners();
        checkHealthStatus();
        elements.chatInput.focus();
    }

    document.addEventListener('DOMContentLoaded', init);

})();
