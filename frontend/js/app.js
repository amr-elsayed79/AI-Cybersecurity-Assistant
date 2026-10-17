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
    let currentInterfaceLang = 'en'; // 'en' or 'ar'

    // DOM Elements
    const elements = {
        // Header & Language Controls
        langBtnAr: document.getElementById('lang-btn-ar'),
        langBtnEn: document.getElementById('lang-btn-en'),
        langToggleBtn: document.getElementById('lang-toggle-btn'),
        langToggleLabel: document.getElementById('lang-toggle-label'),
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
        quickPromptChips: document.querySelectorAll('.quick-prompt-chip')
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
     * Interface string translations for English and Arabic.
     * Technical IDs, citations, and vulnerability names remain untranslated.
     */
    const UI_TRANSLATIONS = {
        en: {
            langButtonLabel: 'العربية',
            langButtonTitle: 'Switch interface to Arabic',
            headerTitle: 'AI Cybersecurity Assistant',
            headerSubtitle: 'Defensive Vulnerability RAG & Attribution Verification',
            mockWarningText: 'MOCK MODE — Workflow testing only; answers are deterministic offline templates and not evidence of real LLM grounding or answer quality.',
            sessionLabel: 'Session ID:',
            freshConversation: 'Fresh Conversation',
            newChatBtn: 'New Chat',
            newChatTitle: 'Clear conversation ID and visible message history',
            welcomeTitle: 'Welcome to the AI Cybersecurity Assistant',
            welcomeDesc: 'This assistant answers defensive software security and web-application vulnerability questions grounded exclusively in the verified Master Knowledge Base (74 canonical vulnerabilities).',
            quickPromptsLabel: 'Try an example question:',
            quickPrompt1: 'Prevent SQL Injection',
            quickPrompt2: 'Reflected vs Stored XSS',
            quickPrompt3: 'حماية من حقن الأوامر (عربي)',
            quickPrompt4: 'Site-Specific Test (SSRF)',
            loadingText: 'Searching Knowledge Base & validating attribution...',
            chatPlaceholder: 'Ask a cybersecurity question... (Enter to send, Shift+Enter for newline)',
            chatInputAria: 'Ask a cybersecurity question',
            sendBtn: 'Send',
            tabEvidence: 'Knowledge Evidence',
            tabDiagnostics: 'System Health',
            turnCardTitle: 'Turn Attribution Status',
            noActiveQuery: 'No Active Query',
            turnDescEmpty: 'Submit a question to inspect retrieved Knowledge Base records, citations, and attribution verification.',
            metricLang: 'Lang:',
            metricRetrieval: 'Retrieval:',
            metricLlm: 'LLM:',
            metricTotal: 'Total:',
            citedTitle: 'Cited Sources',
            citedSubtitle: 'Verified KB chunks referenced in the generated narrative.',
            citedEmpty: 'No cited sources for this turn.',
            retrievedTitle: 'Retrieved Candidates',
            retrievedSubtitle: 'Top chunks retrieved by hybrid search and injected into context.',
            retrievedEmpty: 'No retrieved candidates.',
            diagCardTitle: 'Backend Architecture',
            refreshBtn: 'Refresh',
            diagLabelApiHealth: 'API Health:',
            diagLabelVersion: 'App Version:',
            diagLabelKbHash: 'Master KB Hash:',
            diagLabelVectorStatus: 'Vector Store Status:',
            diagLabelPoints: 'Indexed KB Points:',
            diagLabelLlmProvider: 'Configured LLM:',
            diagLabelLlmCreds: 'LLM Credentials:',
            secNoticeTitle: 'Security & Scope Notice:',
            secNoticeDesc: 'Bound strictly to 127.0.0.1 for local evaluation. Conversation storage is in-process and volatile (lost on server restart). Health checks perform local verification only with zero external network or Groq API calls.'
        },
        ar: {
            langButtonLabel: 'English',
            langButtonTitle: 'تبديل واجهة العرض إلى الإنجليزية',
            headerTitle: 'المساعد الأمني للذكاء الاصطناعي',
            headerSubtitle: 'توليد الإجابات الأمنية والتحقق المباشر من مراجع الثغرات',
            mockWarningText: 'وضع المحاكاة — لاختبار التدفق البرمجي فقط؛ الإجابات قوالب ثابتة وليست دليلاً على أداء النموذج الحقيقي.',
            sessionLabel: 'معرّف الجلسة:',
            freshConversation: 'محادثة جديدة',
            newChatBtn: 'محادثة جديدة',
            newChatTitle: 'مسح معرّف المحادثة وسجل الرسائل المعروض',
            welcomeTitle: 'مرحباً بك في المساعد الأمني للذكاء الاصطناعي',
            welcomeDesc: 'يقدم هذا المساعد إرشادات دفاعية متخصصة في أمن البرمجيات وثغرات تطبيقات الويب استناداً حصرياً إلى قاعدة المعرفة الأمنية المعتمدة (74 ثغرة معتمدة).',
            quickPromptsLabel: 'جرّب أحد الأسئلة المقترحة:',
            quickPrompt1: 'الوقاية من حقن SQL',
            quickPrompt2: 'الفرق بين XSS المنعكس والمخزن',
            quickPrompt3: 'حماية من حقن أوامر النظام',
            quickPrompt4: 'اختبار هدف محدد (SSRF)',
            loadingText: 'جاري البحث في قاعدة المعرفة والتحقق من الأدلة...',
            chatPlaceholder: 'اطرح سؤالاً في الأمن السيبراني... (Enter للإرسال، Shift+Enter لسطر جديد)',
            chatInputAria: 'اطرح سؤالاً في الأمن السيبراني',
            sendBtn: 'إرسال',
            tabEvidence: 'الأدلة والمصادر',
            tabDiagnostics: 'حالة النظام',
            turnCardTitle: 'حالة التحقق من أدلة الإجابة',
            noActiveQuery: 'لا يوجد استعلام نشط',
            turnDescEmpty: 'أرسل سؤالاً لعرض سجلات قاعدة المعرفة المسترجعة، والاستشهادات، والتحقق من الأدلة.',
            metricLang: 'اللغة:',
            metricRetrieval: 'الاسترجاع:',
            metricLlm: 'النموذج:',
            metricTotal: 'الإجمالي:',
            citedTitle: 'المصادر المستشهد بها',
            citedSubtitle: 'مقاطع المعرفة المعتمدة التي تم الاستشهاد بها في الإجابة.',
            citedEmpty: 'لا توجد مصادر مستشهد بها لهذا الرد.',
            retrievedTitle: 'المرشحات المسترجعة',
            retrievedSubtitle: 'أبرز المقاطع المسترجعة بالبحث الهجين والمدرجة في السياق.',
            retrievedEmpty: 'لا توجد مرشحات مسترجعة.',
            diagCardTitle: 'بنية النظام الخلفية',
            refreshBtn: 'تحديث',
            diagLabelApiHealth: 'حالة الواجهة (API):',
            diagLabelVersion: 'إصدار التطبيق:',
            diagLabelKbHash: 'بصمة قاعدة المعرفة:',
            diagLabelVectorStatus: 'حالة محرك المتجهات:',
            diagLabelPoints: 'المقاطع المفهرسة:',
            diagLabelLlmProvider: 'النموذج النشط:',
            diagLabelLlmCreds: 'بيانات الاعتماد:',
            secNoticeTitle: 'تنبيه الأمان والنطاق:',
            secNoticeDesc: 'النظام مقيّد محلياً بعنوان 127.0.0.1 للتقييم الآمن. المحادثات مخزنة في الذاكرة المؤقتة. فحوصات الصحة تجري محلياً دون أي اتصالات خارجية.'
        }
    };

    /**
     * Switch UI interface language between English and Arabic.
     * Preserves active conversation session and current chat feed.
     */
    function setInterfaceLanguage(lang) {
        currentInterfaceLang = (lang === 'ar') ? 'ar' : 'en';
        const isAr = (currentInterfaceLang === 'ar');
        const t = UI_TRANSLATIONS[currentInterfaceLang];

        // 1. Update HTML document direction and language attributes
        document.documentElement.lang = currentInterfaceLang;
        document.documentElement.dir = isAr ? 'rtl' : 'ltr';

        // 2. Update Language Selection Buttons (active class and aria-pressed state)
        if (elements.langBtnAr) {
            if (isAr) {
                elements.langBtnAr.classList.add('active');
                elements.langBtnAr.setAttribute('aria-pressed', 'true');
            } else {
                elements.langBtnAr.classList.remove('active');
                elements.langBtnAr.setAttribute('aria-pressed', 'false');
            }
        }
        if (elements.langBtnEn) {
            if (!isAr) {
                elements.langBtnEn.classList.add('active');
                elements.langBtnEn.setAttribute('aria-pressed', 'true');
            } else {
                elements.langBtnEn.classList.remove('active');
                elements.langBtnEn.setAttribute('aria-pressed', 'false');
            }
        }

        // Legacy Language Toggle Button (fallback)
        if (elements.langToggleLabel) {
            elements.langToggleLabel.textContent = t.langButtonLabel;
        }
        if (elements.langToggleBtn) {
            elements.langToggleBtn.title = t.langButtonTitle;
        }

        // 3. Update Header titles
        const headerTitle = document.querySelector('.header-title');
        if (headerTitle) headerTitle.textContent = t.headerTitle;
        const headerSubtitle = document.querySelector('.header-subtitle');
        if (headerSubtitle) headerSubtitle.textContent = t.headerSubtitle;

        // 4. Update Chat Workspace Chrome
        const sessionLabel = document.querySelector('.session-label');
        if (sessionLabel) sessionLabel.textContent = t.sessionLabel;
        if (!activeConversationId && elements.activeSessionId) {
            elements.activeSessionId.textContent = t.freshConversation;
        }
        if (elements.newChatBtn) {
            const btnText = elements.newChatBtn.lastChild;
            if (btnText && btnText.nodeType === 3) {
                btnText.textContent = ` ${t.newChatBtn}`;
            }
            elements.newChatBtn.title = t.newChatTitle;
        }

        // 5. Update Welcome Card
        if (elements.welcomeMessage) {
            const h2 = elements.welcomeMessage.querySelector('h2');
            if (h2) h2.textContent = t.welcomeTitle;
            const p = elements.welcomeMessage.querySelector('p');
            if (p) p.textContent = t.welcomeDesc;
            const qpLabel = elements.welcomeMessage.querySelector('.quick-prompts-label');
            if (qpLabel) qpLabel.textContent = t.quickPromptsLabel;

            const chips = elements.welcomeMessage.querySelectorAll('.quick-prompt-chip');
            if (chips.length >= 4) {
                chips[0].textContent = t.quickPrompt1;
                chips[1].textContent = t.quickPrompt2;
                chips[2].textContent = t.quickPrompt3;
                chips[3].textContent = t.quickPrompt4;
            }
        }

        // 6. Update Input Form & Controls
        if (elements.chatInput) {
            elements.chatInput.placeholder = t.chatPlaceholder;
            elements.chatInput.setAttribute('aria-label', t.chatInputAria);
        }
        if (elements.sendBtn) {
            const sendSpan = elements.sendBtn.querySelector('span');
            if (sendSpan) sendSpan.textContent = t.sendBtn;
            elements.sendBtn.title = t.sendBtn;
        }
        const loadingText = document.getElementById('loading-text');
        if (loadingText) loadingText.textContent = t.loadingText;

        // 7. Update Mock Banner
        const bannerSpan = elements.mockWarningBanner.querySelector('.banner-text');
        if (bannerSpan) {
            bannerSpan.textContent = '';
            const strongEl = document.createElement('strong');
            strongEl.textContent = isAr ? 'وضع المحاكاة' : 'MOCK MODE';
            bannerSpan.appendChild(strongEl);
            bannerSpan.appendChild(document.createTextNode(isAr
                ? ' — لاختبار التدفق البرمجي فقط؛ الإجابات قوالب ثابتة وليست دليلاً على أداء النموذج الحقيقي.'
                : ' — Workflow testing only; answers are deterministic offline templates and not evidence of real LLM grounding or answer quality.'
            ));
        }
    }

    /**
     * Safely parse inline Markdown (bold, inline code) into a parent DOM element
     * using pure DOM APIs without innerHTML or outerHTML.
     */
    function appendFormattedInline(parent, text) {
        if (!text) return;
        // Matches `code`, **bold**, or __bold__
        const inlineRegex = /(`[^`\n]+`)|(\*\*[^*\n]+?\*\*)|(__[^_\n]+?__)/g;
        let lastIndex = 0;
        let match;

        while ((match = inlineRegex.exec(text)) !== null) {
            if (match.index > lastIndex) {
                parent.appendChild(document.createTextNode(text.slice(lastIndex, match.index)));
            }

            const matchedStr = match[0];
            if (matchedStr.startsWith('`') && matchedStr.endsWith('`')) {
                const code = document.createElement('code');
                code.className = 'inline-code';
                code.textContent = matchedStr.slice(1, -1);
                parent.appendChild(code);
            } else if (
                (matchedStr.startsWith('**') && matchedStr.endsWith('**')) ||
                (matchedStr.startsWith('__') && matchedStr.endsWith('__'))
            ) {
                const strong = document.createElement('strong');
                const innerText = matchedStr.slice(2, -2);
                if (innerText.startsWith('`') && innerText.endsWith('`')) {
                    const code = document.createElement('code');
                    code.className = 'inline-code';
                    code.textContent = innerText.slice(1, -1);
                    strong.appendChild(code);
                } else {
                    strong.textContent = innerText;
                }
                parent.appendChild(strong);
            }

            lastIndex = inlineRegex.lastIndex;
        }

        if (lastIndex < text.length) {
            parent.appendChild(document.createTextNode(text.slice(lastIndex)));
        }
    }

    /**
     * Safely format message content by parsing Markdown blocks (headings, lists, code blocks, paragraphs)
     * and inline formatting using pure DOM APIs without innerHTML or outerHTML.
     */
    function appendSafeContentToBubble(bubbleElement, rawText, isArabic) {
        if (!rawText) return;

        const lines = rawText.split(/\r?\n/);
        let currentList = null;
        let currentParagraph = null;
        let inCodeBlock = false;
        let codeLines = [];
        let codeLang = '';

        function flushParagraph() {
            if (currentParagraph) {
                bubbleElement.appendChild(currentParagraph);
                currentParagraph = null;
            }
        }

        function flushList() {
            if (currentList) {
                bubbleElement.appendChild(currentList.element);
                currentList = null;
            }
        }

        function flushAll() {
            flushParagraph();
            flushList();
        }

        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];

            // Fenced code block toggle
            if (line.trim().startsWith('```')) {
                if (inCodeBlock) {
                    const pre = document.createElement('pre');
                    pre.className = 'code-block-wrapper';
                    const code = document.createElement('code');
                    if (codeLang) {
                        code.className = `language-${codeLang}`;
                    }
                    code.textContent = codeLines.join('\n');
                    pre.appendChild(code);
                    bubbleElement.appendChild(pre);

                    inCodeBlock = false;
                    codeLines = [];
                    codeLang = '';
                } else {
                    flushAll();
                    inCodeBlock = true;
                    codeLines = [];
                    codeLang = line.trim().slice(3).trim();
                }
                continue;
            }

            if (inCodeBlock) {
                codeLines.push(line);
                continue;
            }

            // Blank line separates blocks
            if (!line.trim()) {
                flushAll();
                continue;
            }

            // Headings: # H1, ## H2, ### H3, #### H4
            const headingMatch = line.match(/^(#{1,4})\s+(.+)$/);
            if (headingMatch) {
                flushAll();
                const level = headingMatch[1].length;
                const headingEl = document.createElement(`h${level}`);
                headingEl.className = `md-heading md-h${level}`;
                appendFormattedInline(headingEl, headingMatch[2].trim());
                bubbleElement.appendChild(headingEl);
                continue;
            }

            // Bullet list item: * or -
            const bulletMatch = line.match(/^\s*[-*]\s+(.+)$/);
            if (bulletMatch) {
                flushParagraph();
                if (!currentList || currentList.type !== 'ul') {
                    flushList();
                    currentList = {
                        type: 'ul',
                        element: document.createElement('ul')
                    };
                    currentList.element.className = 'md-list md-bullet-list';
                }
                const li = document.createElement('li');
                appendFormattedInline(li, bulletMatch[1]);
                currentList.element.appendChild(li);
                continue;
            }

            // Numbered list item: 1. or 2.
            const numberMatch = line.match(/^\s*\d+\.\s+(.+)$/);
            if (numberMatch) {
                flushParagraph();
                if (!currentList || currentList.type !== 'ol') {
                    flushList();
                    currentList = {
                        type: 'ol',
                        element: document.createElement('ol')
                    };
                    currentList.element.className = 'md-list md-numbered-list';
                }
                const li = document.createElement('li');
                appendFormattedInline(li, numberMatch[1]);
                currentList.element.appendChild(li);
                continue;
            }

            // Indented continuation of list item
            if (currentList && /^\s{2,}\S/.test(line)) {
                const lastLi = currentList.element.lastElementChild;
                if (lastLi) {
                    lastLi.appendChild(document.createElement('br'));
                    appendFormattedInline(lastLi, line.trim());
                    continue;
                }
            }

            // Regular paragraph line
            flushList();
            if (!currentParagraph) {
                currentParagraph = document.createElement('p');
                currentParagraph.className = 'md-paragraph';
            } else {
                currentParagraph.appendChild(document.createElement('br'));
            }
            appendFormattedInline(currentParagraph, line);
        }

        // Handle unclosed code block gracefully
        if (inCodeBlock && codeLines.length > 0) {
            const pre = document.createElement('pre');
            pre.className = 'code-block-wrapper';
            const code = document.createElement('code');
            if (codeLang) code.className = `language-${codeLang}`;
            code.textContent = codeLines.join('\n');
            pre.appendChild(code);
            bubbleElement.appendChild(pre);
        }

        flushAll();
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
        const isAr = (currentInterfaceLang === 'ar');
        switch (status) {
            case 'grounded':
                return isAr ? 'موثق (أدلة معتمدة)' : 'Grounded (Verified KB)';
            case 'site_qualified':
                return isAr ? 'مشروط بفحص الهدف' : 'Site-Specific Qualified';
            case 'insufficient_evidence':
                return isAr ? 'أدلة غير كافية' : 'Insufficient KB Evidence';
            case 'empty_retrieval':
                return isAr ? 'لم يتم العثور على أدلة' : 'No Relevant Evidence Found';
            case 'attribution_failed':
                return isAr ? 'محجوب لعدم تطابق الأدلة' : 'Attribution Check Suppressed';
            case 'incomplete_truncated':
                return isAr ? 'إجابة مجتزأة / تجاوز الحد' : 'Incomplete / Token Truncated';
            default:
                return status || (isAr ? 'مكتمل' : 'Completed');
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
    /**
     * Update the Evidence panel for the active turn (panel removed from UI).
     */
    function updateEvidencePanel(response) {
        // Evidence and diagnostics panel removed from UI; no-op
    }

    /**
     * Clear active conversation state and reset UI (New Chat action).
     */
    function resetConversation() {
        activeConversationId = null;
        if (elements.activeSessionId) {
            elements.activeSessionId.textContent = 'Fresh Conversation';
            elements.activeSessionId.title = 'No active conversation ID yet';
        }

        // Reset chat feed
        elements.chatMessages.textContent = '';
        if (elements.welcomeMessage) {
            elements.welcomeMessage.style.display = 'block';
            elements.chatMessages.appendChild(elements.welcomeMessage);
        }

        // Hide any active error or status
        elements.chatStatusBar.classList.add('hidden');
        elements.errorAlert.classList.add('hidden');
        elements.chatInput.value = '';
        elements.charCounter.textContent = '0 / 2000';
        elements.chatInput.focus();
    }

    /**
     * Check backend health and update provider banner if in mock mode.
     */
    async function checkHealthStatus() {
        try {
            const result = await window.AssistantAPI.getHealth();
            if (result && result.data) {
                configuredProvider = (result.data.llm_provider || 'unknown').toLowerCase();
                if (elements.mockWarningBanner) {
                    if (configuredProvider === 'mock') {
                        elements.mockWarningBanner.classList.remove('hidden');
                    } else {
                        elements.mockWarningBanner.classList.add('hidden');
                    }
                }
            }
        } catch (err) {
            // Gracefully ignore local network failures
        }
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
                if (elements.activeSessionId) {
                    elements.activeSessionId.textContent = activeConversationId;
                    elements.activeSessionId.title = `Active in-memory session: ${activeConversationId}`;
                }
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
     * Tab switching logic (Evidence and diagnostics panel removed from UI).
     */
    function setupTabs() {
        // Evidence and diagnostics panel removed from UI; no-op
    }

    /**
     * Setup event listeners for forms, buttons, and keyboard controls.
     */
    function setupEventListeners() {
        // Language Selection Buttons (explicit Arabic / English)
        if (elements.langBtnAr) {
            elements.langBtnAr.addEventListener('click', () => {
                setInterfaceLanguage('ar');
            });
        }
        if (elements.langBtnEn) {
            elements.langBtnEn.addEventListener('click', () => {
                setInterfaceLanguage('en');
            });
        }

        // Language Toggle Button (switches interface language between en and ar)
        if (elements.langToggleBtn) {
            elements.langToggleBtn.addEventListener('click', () => {
                const nextLang = (currentInterfaceLang === 'en') ? 'ar' : 'en';
                setInterfaceLanguage(nextLang);
            });
        }

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
