const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");

function addMessage(text, className) {
    const message = document.createElement("div");
    message.classList.add("message", className);
    message.innerText = text;
    chatBox.appendChild(message);
    chatBox.scrollTop = chatBox.scrollHeight;
}

async function sendMessage() {
    const question = userInput.value.trim();
    if (!question) return;

    addMessage(question, "user-message");
    userInput.value = "";

    addMessage("Typing...", "bot-message");

    try {
        const response = await fetch("http://127.0.0.1:8000/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ question })
        });

        const data = await response.json();

        // Remove "Typing..."
        chatBox.lastChild.remove();

        addMessage(data.answer || "No response received.", "bot-message");

    } catch (error) {
        chatBox.lastChild.remove();
        addMessage("Error connecting to server.", "bot-message");
    }
}

sendBtn.addEventListener("click", sendMessage);

userInput.addEventListener("keypress", function (e) {
    if (e.key === "Enter") {
        sendMessage();
    }
});