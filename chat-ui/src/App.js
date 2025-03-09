import React, { useState, useRef, useEffect } from 'react';
import { Send } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import './index.css';

// Uncomment on deploy
// const BACKEND_ROUTE = "api/routes/chat/";
// const EXTRACTION_ROUTE = "extraction";
// Comment on deploy
const BACKEND_ROUTE = "http://localhost:8080/api/routes/chat/";
// Bad bad, skeptical this will work when we change it to the other route
const EXTRACTION_ROUTE ="http://127.0.0.1:8080/extraction"; 


const ChatInterface = () => {
  const [messages, setMessages] = useState([
    {
      text: "Hi, I'm here to help with any questions about Flare! What would you like to know?",
      type: 'bot'
    }
  ]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isExtracting, setIsExtracting] = useState(false);
  const [awaitingConfirmation, setAwaitingConfirmation] = useState(false);
  const [pendingTransaction, setPendingTransaction] = useState(null);
  const messagesEndRef = useRef(null);

  const [doc1, setdoc1] = useState('');
  const [doc2, setdoc2] = useState('');
  const [doc3, setdoc3] = useState('');
  const [doc4, setdoc4] = useState('');
  const [doc5, setdoc5] = useState('');

  const [topDocuments, setTopDocuments] = useState([
      {
          file_name: "",
          score: "", 
      }
  ]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (text) => {
    try {
      const response = await fetch(BACKEND_ROUTE, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: text }),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();

      // Check if response contains a transaction preview
      if (data.response.includes('Transaction Preview:')) {
        setAwaitingConfirmation(true);
        setPendingTransaction(text);
      }
      const doc_scores_dict = JSON.parse(data.doc_scores);
      const keys = Object.keys(doc_scores_dict);
        setdoc1("[" + keys[0] + "]" + "\n" + "Similarity Score: " + doc_scores_dict[keys[0]])
        setdoc2("[" + keys[1] + "]" + "\n" + "Similarity Score: " + doc_scores_dict[keys[1]])
        setdoc3("[" + keys[2] + "]" + "\n" + "Similarity Score: " + doc_scores_dict[keys[2]])
        setdoc4("[" + keys[3] + "]" + "\n" + "Similarity Score: " + doc_scores_dict[keys[3]])
        setdoc5("[" + keys[4] + "]" + "\n" + "Similarity Score: " + doc_scores_dict[keys[4]])

      return data.response;
    } catch (error) {
      console.error('Error:', error);
      return 'Sorry, there was an error processing your request. Please try again.';
    }
  };

  const [extraction_data, set_extraction_data] = useState({});
  const [extraction_key, set_extraction_key] = useState("");
  const [extraction_value, set_extraction_value] = useState("");
  const [scrape_urls, set_scrape_urls] = useState("");
  const [use_llm, set_use_llm] = useState(false);

  const handleRunExtraction = async () => {
      setIsExtracting(true);
    try {
      const response = await fetch(EXTRACTION_ROUTE, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(extraction_data),
      });

      if (!response.ok) {
          setIsExtracting(false);
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
          setIsExtracting(false);
      return data.response;
    } catch (error) {
      console.error('Error:', error);
          setIsExtracting(false);
      return 'Sorry, there was an error processing your request. Please try again.';
    }
  };

  const handleScrapeUpdateExtractionPipeline = async (e) => {
      var new_data = {"urls":e}
      new_data['use_llm'] = use_llm;
      set_extraction_data(extraction_data=> ({
        ...extraction_data,  // Spread the previous dictionary
        ['scrape']: new_data // Add or update the new key-value pair
      }));
      console.log(extraction_data);
  }
  const handleCrawlUpdateExtractionPipeline = async (e) => {
      var depth = document.getElementById("crawl_depth");
      var value = depth.value;
      var text = depth.options[depth.selectedIndex].text;
      var num = parseInt(text);
      var new_data = {"urls":e}
      new_data['use_llm'] = use_llm;
      new_data['max_pages'] = num;
      set_extraction_data(extraction_data=> ({
        ...extraction_data,  // Spread the previous dictionary
        ['web_crawl']: new_data // Add or update the new key-value pair
      }));
      console.log(extraction_data);
  }

    const textArea = document.querySelector('textarea');
    if (textArea) {
        textArea.addEventListener('change', (e) => {
          // Process the entire text without splitting by newlines
          handleScrapeUpdateExtractionPipeline(e.target.value);
        });
    }
    const input = document.querySelector('input[name="crawl_input"]')
    if (input) {
        input.addEventListener('change', (e) => {
          // Process the entire text without splitting by newlines
          handleCrawlUpdateExtractionPipeline(e.value);
        });
    }

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputText.trim() || isLoading) return;

    const messageText = inputText.trim();
    setInputText('');
    setIsLoading(true);
    setMessages(prev => [...prev, { text: messageText, type: 'user' }]);

    // Handle transaction confirmation
    if (awaitingConfirmation) {
      if (messageText.toUpperCase() === 'CONFIRM') {
        setAwaitingConfirmation(false);
        const response = await handleSendMessage(pendingTransaction);
        setMessages(prev => [...prev, { text: response, type: 'bot' }]);
      } else {
        setAwaitingConfirmation(false);
        setPendingTransaction(null);
        setMessages(prev => [...prev, {
          text: 'Transaction cancelled. How else can I help you?',
          type: 'bot'
        }]);
      }
    } else {
      const response = await handleSendMessage(messageText);
      setMessages(prev => [...prev, { text: response, type: 'bot' }]);
    }

    setIsLoading(false);
  };


  // Custom components for ReactMarkdown
  const MarkdownComponents = {
    // Override paragraph to remove default margins
    p: ({ children }) => <span className="inline">{children}</span>,
    // Style code blocks
    code: ({ node, inline, className, children, ...props }) => (
      inline ?
        <code className="bg-gray-200 rounded px-1 py-0.5 text-sm">{children}</code> :
        <pre className="bg-gray-200 rounded p-2 my-2 overflow-x-auto">
          <code {...props} className="text-sm">{children}</code>
        </pre>
    ),
    // Style links
    a: ({ node, children, ...props }) => (
      <a {...props} className="text-pink-600 hover:underline">{children}</a>
    )
  };

  return (
    <div className="flex flex-col h-screen bg-gray-100">
      <div class='top_document'>
      <p> Top documents from most recent query: </p>
      <p> {doc1} </p>
      <p> {doc2} </p>
      <p> {doc3} </p>
      </div>
      <div className="flex flex-col h-full max-w-4xl mx-auto w-full shadow-lg bg-white">
        {/* Header */}
        <div className="bg-pink-600 text-white p-4">
          <h1 className="text-xl font-bold">Flare AI RAG</h1>
          <p className="text-sm opacity-80">(Based on Flare Dev Hub)</p>
        </div>

        {/* Messages container */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.map((message, index) => (
            <div
              key={index}
              className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              {message.type === 'bot' && (
                <div className="w-8 h-8 rounded-full bg-pink-600 flex items-center justify-center text-white font-bold mr-2">
                  A
                </div>
              )}
              <div
                className={`max-w-xs px-4 py-2 rounded-xl ${
                  message.type === 'user'
                    ? 'bg-pink-600 text-white rounded-br-none'
                    : 'bg-gray-100 text-gray-800 rounded-bl-none'
                }`}
              >
                <ReactMarkdown
                  components={MarkdownComponents}
                  className="text-sm break-words whitespace-pre-wrap"
                >
                  {message.text}
                </ReactMarkdown>
              </div>
              {message.type === 'user' && (
                <div className="w-8 h-8 rounded-full bg-gray-400 flex items-center justify-center text-white font-bold ml-2">
                  U
                </div>
              )}
            </div>
          ))}
          {isLoading && (
            <div className="flex justify-start">
              <div className="w-8 h-8 rounded-full bg-pink-600 flex items-center justify-center text-white font-bold mr-2">
                A
              </div>
              <div className="bg-gray-100 text-gray-800 px-4 py-2 rounded-xl rounded-bl-none">
                <div className="flex space-x-2">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input form */}
        <div className="border-t border-gray-200 p-4">
          <form onSubmit={handleSubmit} className="flex space-x-4">
            <input
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder={awaitingConfirmation ? "Type CONFIRM to proceed or anything else to cancel" : "Type your message... (Markdown supported)"}
              className="flex-1 px-4 py-2 border border-gray-300 rounded-full focus:outline-none focus:ring-2 focus:ring-pink-500 focus:border-transparent"
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={isLoading}
              className="bg-pink-600 text-white p-2 rounded-full hover:bg-pink-700 focus:outline-none focus:ring-2 focus:ring-pink-500 focus:ring-offset-2 disabled:opacity-50"
            >
              <Send className="w-5 h-5" />
            </button>
          </form>
        </div>
      <div class='extraction_dashboard'>
      <p> Extraction Dashboard </p>
      <p> Database last updated: </p>
      <p name = "last_updated_label" value = ""> </p>
      
      <input type="checkbox" id="scrape_input" name="scrape_input" />
      <label for="vehicle1"> Preprocess Extractions with gemini-1.5-flash</label>
      <p> </p>
      <label for="scrape_input"> Input urls to scrape into vector database </label>
      <textarea name="textArea" cols="30" rows="5"  onChange={(e) => handleScrapeUpdateExtractionPipeline(e.target.value)} ></textarea>
      <select name="crawl_depth" id="crawl_depth">
          <option value="1">Crawl 1 webpage deep</option>
          <option value="5">Crawl 5 webpages deep</option>
          <option value="10">Crawl 10 webpages deep</option>
          <option value="30">Crawl 30 webpages deep</option>
        </select>
      <p> </p>
      <label for="crawl_input"> Input a single url to crawl </label>
      <input type="text" class = "crawl_input" id="crawl_input" onChange={(e) => handleCrawlUpdateExtractionPipeline(e.target.value)} name="crawl_input" />
      <p> </p>
          <div class="database_stats">
          <label for="database_data" value =""> Database statistics: </label>
          <p class = "wepbage_stat" value = ""> webpages:  </p>
          <p class = ".mdx_stat" value = ""> mdx files:  </p>
          </div>
      <button onClick={handleRunExtraction} disabled={isExtracting}> Run extraction {isExtracting ? 'Running Extraction Pipeline...' : '' } </button>
      </div>
      </div>
    </div>
  );
};

export default ChatInterface;
