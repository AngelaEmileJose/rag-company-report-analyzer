document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const tabs = document.querySelectorAll('.nav-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    const urlInput = document.getElementById('url-input');
    const processBtn = document.getElementById('process-url-btn');
    const dropZone = document.getElementById('drop-zone');
    const processingStatus = document.getElementById('processing-status');
    const statusText = document.getElementById('status-text');
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const chatHistory = document.getElementById('chat-history');
    const generateBtn = document.getElementById('generate-btn');
    const analysisResults = document.getElementById('analysis-results');

    // Status Panel
    const docInfoPanel = document.getElementById('doc-info-panel');
    const docNameDisplay = document.getElementById('current-doc-name');
    const docChunksDisplay = document.getElementById('doc-chunks');
    const docTimeDisplay = document.getElementById('doc-time');
    const systemStatus = document.getElementById('system-status');
    const statusDot = document.querySelector('.status-dot');

    // UI Configuration
    let processed = false;
    let currentCompany = "";

    // -- Tab Switching Logic --
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.disabled) return;

            // Remove active class from all
            tabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));

            // Add active class to clicked
            tab.classList.add('active');

            // Show content
            const tabId = tab.dataset.tab;
            document.getElementById(`${tabId}-tab`).classList.add('active');
        });
    });

    const fileInput = document.getElementById('file-input'); // Select file input

    // -- File Upload Logic --
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            uploadFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            uploadFile(e.target.files[0]);
        }
    });

    async function uploadFile(file) {
        if (file.type !== 'application/pdf') {
            alert('Please upload a PDF file.');
            return;
        }

        setProcessing(true);
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                processed = true;
                updateDocInfo(data.filename, data.chunks, data.time);
                enableTabs();
                addSystemMessage(`Successfully uploaded and processed ${data.filename}. ${data.chunks} chunks created in ${data.time.toFixed(2)}s.`);
                document.querySelector('[data-tab="chat"]').click();
            } else {
                alert(`Error: ${data.detail || 'Upload failed'}`);
            }
        } catch (error) {
            console.error(error);
            alert("Network error uploading file");
        } finally {
            setProcessing(false);
            fileInput.value = ''; // Reset
        }
    }

    // -- Document Processing (URL) --
    async function processDocument(source) {
        setProcessing(true);
        try {
            const response = await fetch('/process', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ source: source, company: "Detect" })
            });

            const data = await response.json();

            if (response.ok) {
                // Success
                processed = true;
                updateDocInfo(data.source, data.chunks, data.time);
                enableTabs();
                addSystemMessage(`Successfully processed document. ${data.chunks} text chunks created in ${data.time.toFixed(2)}s.`);

                // Switch to chat
                document.querySelector('[data-tab="chat"]').click();
            } else {
                alert(`Error: ${data.detail || 'Processing failed'}`);
            }
        } catch (error) {
            console.error(error);
            alert("Network error processing document");
        } finally {
            setProcessing(false);
        }
    }

    processBtn.addEventListener('click', () => {
        const url = urlInput.value.trim();
        if (url) {
            processDocument(url);
        }
    });

    // -- Chat Logic --
    async function sendMessage() {
        const question = chatInput.value.trim();
        if (!question || !processed) return;

        // Add user message
        addMessage(question, 'user');
        chatInput.value = '';

        // Add loading placeholder
        const loadingId = addMessage('Thinking...', 'system', true);

        try {
            const response = await fetch('/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: question })
            });

            const data = await response.json();

            // Remove loading, add answer
            removeMessage(loadingId);
            addMessage(data.answer, 'system');

        } catch (error) {
            removeMessage(loadingId);
            addMessage("Sorry, I encountered an error getting the answer.", 'system');
        }
    }

    sendBtn.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // -- Analysis Logic --
    const summaryBtn = document.getElementById('summary-btn');
    const summaryResults = document.getElementById('summary-results');

    summaryBtn.addEventListener('click', async () => {
        summaryResults.innerHTML = '<div class="spinner"></div> Generating executive briefing...';
        summaryResults.classList.remove('hidden');

        try {
            const response = await fetch('/summary', { method: 'POST' });
            const data = await response.json();

            if (data.summary) {
                renderSummary(data.summary);
            } else {
                summaryResults.innerHTML = 'Failed to generate summary.';
            }
        } catch (e) {
            summaryResults.innerHTML = 'Error generating summary.';
        }
    });

    function renderSummary(items) {
        let html = `<h3>Executive Summary</h3><div class="summary-list">`;
        items.forEach(item => {
            html += `
                <div class="summary-item">
                    <h4><i class="fa-solid fa-star" style="color:var(--primary)"></i> ${item.topic}</h4>
                    <p class="summary-q">${item.question}</p>
                    <div class="summary-a">${item.answer.replace(/\n/g, '<br>')}</div>
                </div>
                <hr style="border:0; border-top:1px solid var(--border); margin: 20px 0;">
            `;
        });
        html += '</div>';
        summaryResults.innerHTML = html;
    }

    generateBtn.addEventListener('click', async () => {
        const company = document.getElementById('company-name').value || "the company";
        const topic = document.getElementById('analysis-topic').value || "General Analysis";

        analysisResults.innerHTML = '<div class="spinner"></div> Generating analysis...';
        analysisResults.classList.remove('hidden');

        try {
            const response = await fetch('/generate_questions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ company, topic, count: 5 })
            });

            const data = await response.json();

            if (data.questions && data.questions.length > 0) {
                renderAnalysis(data.questions, topic);
            } else {
                analysisResults.innerHTML = 'No questions generated. Try a different topic.';
            }
        } catch (e) {
            analysisResults.innerHTML = 'Error generating analysis.';
        }
    });

    function renderAnalysis(questions, topic) {
        let html = `<h3>Deep Dive: ${topic}</h3><ul class="analysis-list">`;
        questions.forEach(q => {
            html += `
                <li class="analysis-item">
                    <span class="q-text">${q}</span>
                    <button class="ask-btn" onclick="askAnalysisQuestion('${q.replace(/'/g, "\\'")}')">
                        Ask <i class="fa-solid fa-arrow-right"></i>
                    </button>
                </li>
            `;
        });
        html += '</ul>';
        analysisResults.innerHTML = html;
    }

    // Expose for onclick
    window.askAnalysisQuestion = (q) => {
        document.querySelector('[data-tab="chat"]').click();
        chatInput.value = q;
        sendMessage();
    };

    // -- Helpers --
    function setProcessing(isProcessing) {
        if (isProcessing) {
            processingStatus.classList.remove('hidden');
            processBtn.disabled = true;
            statusText.innerText = "Processing document...";
        } else {
            processingStatus.classList.add('hidden');
            processBtn.disabled = false;
        }
    }

    function updateDocInfo(source, chunks, time) {
        docInfoPanel.style.display = 'block';
        docNameDisplay.innerText = source.split('/').pop().substring(0, 20) + '...';
        docNameDisplay.title = source;
        docChunksDisplay.innerText = chunks;
        docTimeDisplay.innerText = time.toFixed(1) + 's';

        systemStatus.innerText = "Document Loaded";
        statusDot.style.backgroundColor = "#00b894";
        statusDot.style.boxShadow = "0 0 10px #00b894";
    }

    function enableTabs() {
        document.getElementById('chat-tab-btn').disabled = false;
        document.getElementById('analyze-tab-btn').disabled = false;
    }

    function addMessage(text, type, isLoading = false) {
        const div = document.createElement('div');
        div.className = `message ${type}`;
        div.id = isLoading ? 'msg-loading' : 'msg-' + Date.now();

        // Convert Markdown-ish bold to bold tag
        let formatted = text.replace(/\*\*(.*?)\*\*/g, '<b>$1</b>');
        formatted = formatted.replace(/\n/g, '<br>');

        div.innerHTML = `<div class="message-content">${formatted}</div>`;
        chatHistory.appendChild(div);
        chatHistory.scrollTop = chatHistory.scrollHeight;
        return div.id;
    }

    function removeMessage(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    function addSystemMessage(text) {
        addMessage(text, 'system');
    }
});
