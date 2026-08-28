/**
 * CLANKER — Intelligent Skeuomorphic Notebook AI Assistant
 * Dual-Path Routing (RAG Knowledge Mode vs Casual Conversation Mode)
 */

document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const landingView = document.getElementById('landing-view');
  const chatView = document.getElementById('chat-view');
  const btnEnterChat = document.getElementById('btn-enter-chat');
  const btnBackCover = document.getElementById('btn-back-cover');
  const btnClearChat = document.getElementById('btn-clear-chat');
  const chatMessages = document.getElementById('chat-messages');
  const chatForm = document.getElementById('chat-form');
  const questionInput = document.getElementById('question-input');
  const btnSubmit = document.getElementById('btn-submit-question');
  const loadingIndicator = document.getElementById('loading-indicator');
  const loadingMainText = loadingIndicator.querySelector('.loading-main');
  const loadingSubText = loadingIndicator.querySelector('.loading-sub');
  const docStatusIndicator = document.getElementById('doc-status-indicator');

  // In-Memory Multi-Turn Conversation History
  let conversationHistory = [];

  // Fetch document metadata on initial load
  fetchDocumentStatus();

  // Navigation: Enter Notebook Chat
  btnEnterChat.addEventListener('click', () => {
    landingView.classList.add('hidden');
    chatView.classList.remove('hidden');
    questionInput.focus();
  });

  // Navigation: Return to Cover
  btnBackCover.addEventListener('click', () => {
    chatView.classList.add('hidden');
    landingView.classList.remove('hidden');
  });

  // Clear Chat History
  btnClearChat.addEventListener('click', () => {
    const welcome = document.getElementById('welcome-message');
    chatMessages.innerHTML = '';
    conversationHistory = [];
    if (welcome) {
      chatMessages.appendChild(welcome);
    }
    questionInput.focus();
  });

  // Quick Prompt Chips
  document.addEventListener('click', (e) => {
    const chip = e.target.closest('.prompt-chip');
    if (chip) {
      const promptText = chip.getAttribute('data-prompt');
      if (promptText) {
        questionInput.value = promptText;
        submitQuery(promptText);
      }
    }
  });

  // Keyboard Submission in Textarea (Enter to submit, Shift+Enter for newline)
  questionInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      chatForm.requestSubmit();
    }
  });

  // Form Submit Handler
  chatForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const queryText = questionInput.value.trim();
    if (!queryText) return;
    submitQuery(queryText);
  });

  async function fetchDocumentStatus() {
    try {
      const res = await fetch('/api/documents');
      if (res.ok) {
        const data = await res.json();
        const docCount = data.documents ? data.documents.length : 3;
        const chunkCount = data.total_chunks || 0;
        docStatusIndicator.textContent = `Indexed: ${docCount} NimbusNote Documents (${chunkCount} Chunks)`;
      }
    } catch (err) {
      console.warn('Could not fetch document status:', err);
    }
  }

  async function submitQuery(question) {
    // Clear input & disable button
    questionInput.value = '';
    btnSubmit.disabled = true;

    // Append User Message to Notebook
    appendUserMessage(question);

    // Contextual Loading State
    const isCasual = /^(yo|hey|hi|hello|sup|thanks|lol|tell me a joke|who are you|how are you)/i.test(question.trim());
    if (isCasual) {
      loadingMainText.textContent = 'Clanker is thinking...';
      loadingSubText.textContent = 'Writing a conversational note';
    } else {
      loadingMainText.textContent = 'Clanker is checking the notebook...';
      loadingSubText.textContent = 'Searching indexed passages & calculating similarity';
    }

    showLoading(true);
    scrollToBottom();

    // Prepare payload with recent history
    const historyPayload = conversationHistory.slice(-6);

    try {
      const response = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: question,
          history: historyPayload
        })
      });

      if (!response.ok) {
        throw new Error(`Server returned status ${response.status}`);
      }

      const data = await response.json();
      showLoading(false);

      // Append assistant turn to history
      conversationHistory.push({ role: 'user', content: question });
      conversationHistory.push({ role: 'assistant', content: data.answer });

      appendAssistantResponse(data);
    } catch (error) {
      console.error('Query processing error:', error);
      showLoading(false);
      appendErrorMessage('An error occurred while writing in the notebook. Please try again.');
    } finally {
      btnSubmit.disabled = false;
      questionInput.focus();
      scrollToBottom();
    }
  }

  function appendUserMessage(text) {
    const article = document.createElement('article');
    article.className = 'chat-message user-message';
    article.innerHTML = `
      <div class="speaker-tag">
        <svg class="speaker-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
          <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
          <circle cx="12" cy="7" r="4"></circle>
        </svg>
        <span>YOU</span>
      </div>
      <div class="message-body handwriting-text">
        <p>${escapeHtml(text)}</p>
      </div>
    `;
    chatMessages.appendChild(article);
  }

  function appendAssistantResponse(data) {
    const { mode, answer, citations, top_similarity, threshold } = data;
    const article = document.createElement('article');
    article.className = 'chat-message assistant-message';

    let retrievalBoxHtml = '';
    let citationsFooterHtml = '';

    // PATH 1: RAG KNOWLEDGE MODE (With Visible Retrieval Card & Citations)
    if (mode === 'rag' && citations && citations.length > 0) {
      retrievalBoxHtml = `
        <div class="retrieval-box" role="region" aria-label="Retrieved Document Passage">
          <div class="retrieval-header">
            <div class="retrieval-title-group">
              <svg class="retrieval-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                <polyline points="14 2 14 8 20 8"></polyline>
                <line x1="16" y1="13" x2="8" y2="13"></line>
                <line x1="16" y1="17" x2="8" y2="17"></line>
                <polyline points="10 9 9 9 8 9"></polyline>
              </svg>
              <span>RETRIEVED FROM NOTEBOOK</span>
            </div>
            <div class="retrieval-meta-badges">
              <span class="doc-badge">📎 ${escapeHtml(citations[0].source)}</span>
              <span class="section-badge">Section: ${escapeHtml(citations[0].section)}</span>
              <span class="score-badge">Similarity: ${citations[0].similarity.toFixed(2)}</span>
            </div>
          </div>
          <div class="passage-quote-container">
            <span class="passage-label">Retrieved Passage:</span>
            <blockquote class="passage-quote">
              "${escapeHtml(citations[0].passage)}"
            </blockquote>
          </div>
        </div>
      `;

      const tags = citations.map(c => 
        `<span class="citation-tag"><strong>Source:</strong> ${escapeHtml(c.source)} (<strong>${escapeHtml(c.section)}</strong>, sim: ${c.similarity.toFixed(2)})</span>`
      ).join(' ');

      citationsFooterHtml = `
        <div class="citation-footer" aria-label="Source Document Metadata">
          ${tags}
        </div>
      `;
    } 
    // PATH 2: UNSUPPORTED FACTUAL QUERY (Calm Not-Found Presentation)
    else if (mode === 'unsupported') {
      retrievalBoxHtml = `
        <div class="calm-unsupported-box" role="region" aria-label="Information Not Found">
          <div class="calm-unsupported-header">
            <svg class="speaker-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
              <circle cx="12" cy="12" r="10"></circle>
              <line x1="12" y1="8" x2="12" y2="12"></line>
              <line x1="12" y1="16" x2="12.01" y2="16"></line>
            </svg>
            <span>NOT FOUND IN NOTEBOOK</span>
          </div>
          <p class="calm-unsupported-desc">
            No matching information was found in the NimbusNote documents (similarity: <strong>${top_similarity ? top_similarity.toFixed(2) : '0.00'}</strong> &lt; threshold: <strong>${threshold ? threshold.toFixed(2) : '0.40'}</strong>).
          </p>
        </div>
      `;
    }
    // PATH 3: CASUAL CONVERSATION MODE (No retrieval cards or error boxes)

    article.innerHTML = `
      <div class="speaker-tag">
        <img src="/assets/clanker-mascot.svg" alt="" class="speaker-avatar-img" width="16" height="16" aria-hidden="true">
        <span>CLANKER</span>
      </div>

      <!-- Retrieval Box (if in RAG mode) or Calm Notice (if unsupported) -->
      ${retrievalBoxHtml}

      <!-- Conversational Answer in Handwriting Typography -->
      <div class="message-body handwriting-text">
        <p>${escapeHtml(answer)}</p>
      </div>

      <!-- Source Citations (if in RAG mode) -->
      ${citationsFooterHtml}
    `;

    chatMessages.appendChild(article);
  }

  function appendErrorMessage(errorText) {
    const article = document.createElement('article');
    article.className = 'chat-message assistant-message';
    article.innerHTML = `
      <div class="speaker-tag">
        <img src="/assets/clanker-mascot.svg" alt="" class="speaker-avatar-img" width="16" height="16" aria-hidden="true">
        <span>CLANKER</span>
      </div>
      <div class="calm-unsupported-box">
        <div class="calm-unsupported-header">Notebook Notice</div>
        <p class="calm-unsupported-desc">${escapeHtml(errorText)}</p>
      </div>
    `;
    chatMessages.appendChild(article);
  }

  function showLoading(show) {
    if (show) {
      loadingIndicator.classList.remove('hidden');
    } else {
      loadingIndicator.classList.add('hidden');
    }
  }

  function scrollToBottom() {
    setTimeout(() => {
      chatMessages.scrollTop = chatMessages.scrollHeight;
    }, 50);
  }

  function escapeHtml(text) {
    if (!text) return '';
    return text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }
});
