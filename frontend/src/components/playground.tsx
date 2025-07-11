import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import userAvatar from "@/assets/user-avatar.png";
import assistantAvatar from "@/assets/assistant-avatar.png"

import {
  useChatInteract,
  useChatMessages,
  IStep,
} from "@chainlit/react-client";
import { useMemo, useState } from "react";

function flattenMessages(
  messages: IStep[], 
  condition: (node: IStep) => boolean
): IStep[] {
  return messages.reduce((acc: IStep[], node) => {
    if (condition(node)) {
      acc.push(node);
    }
    
    if (node.steps?.length) {
      acc.push(...flattenMessages(node.steps, condition));
    }
    
    return acc;
  }, []);
}

export function Playground() {
  const [inputValue, setInputValue] = useState("");
  const { sendMessage } = useChatInteract();
  const { messages } = useChatMessages();

  const flatMessages = useMemo(() => {
    return flattenMessages(messages, (m) => m.type.includes("message"))
  }, [messages])

  const handleSendMessage = () => {
    const content = inputValue.trim();
    if (content) {
      const message = {
        name: "user",
        type: "user_message" as const,
        output: content,
      };
      sendMessage(message, []);
      setInputValue("");
    }
  };

  const renderMessage = (message: IStep) => {
    const isUser = message.type === "user_message";
    const date = new Date(message.createdAt).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    });

    // pick the right avatar
    const avatarSrc = isUser ? userAvatar : assistantAvatar;
    const bubbleBg = isUser
      ? "bg-blue-100 dark:bg-blue-900"
      : "bg-white dark:bg-gray-800";
    const justify = isUser ? "justify-end" : "justify-start";
    const imgMargin = isUser ? "ml-2" : "mr-2";

    return (
      <div key={message.id} className={`flex items-start mb-4 ${justify}`}>
        {/* Avatar */}
        <img
          src={avatarSrc}
          alt={isUser ? "You" : "Assistant"}
          className={`w-8 h-8 rounded-full ${imgMargin}`}
        />

        {/* Message bubble */}
        <div className={`max-w-[70%] border rounded-lg p-2 ${bubbleBg}`}>
          <p className={`text-black dark:text-white ${isUser && "text-right"}`}>
            {message.output}
          </p>
          <small className="text-xs text-gray-500 block">
            {isUser ? (
              <span className="text-right block">{date}</span>
            ) : (
              date
            )}
          </small>
        </div>
      </div>
    );
  };
  
   return (
    <div className="flex h-screen bg-gray-100 dark:bg-gray-900">
      
      {/* LEFT SIDEBAR */}
      <aside className="w-3/4 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 p-4 overflow-auto">
        <h2 className="text-lg font-semibold mb-4">Components</h2>
        {/* 🚀 Drop your custom components here */}
        {/* <MyCustomControl /> */}
      </aside>

      {/* RIGHT CHAT PANEL */}
      <div className="flex flex-col flex-1">
        {/* message list */}
        <div className="flex-1 overflow-auto p-6">
          <div className="space-y-4">
            {flatMessages.map(renderMessage)}
          </div>
        </div>

        {/* input area */}
        <div className="border-t p-4 bg-white dark:bg-gray-800">
          <div className="flex items-center space-x-2">
            <Input
              autoFocus
              className="flex-1"
              id="message-input"
              placeholder="Type a message"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyUp={(e) => {
                if (e.key === "Enter") handleSendMessage();
              }}
            />
            <Button onClick={handleSendMessage} type="submit">
              Send
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

