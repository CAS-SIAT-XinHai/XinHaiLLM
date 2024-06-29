<template>
  <div>
    <a-row>
      <a-col :span="24">
        <a-select
            :loading="loading"
            v-model="modelName"
            v-model:options="modelOptions"></a-select>
      </a-col>
    </a-row>
    <vue-advanced-chat
        height="calc(100vh - 20px)"
        :current-user-id="currentUserId"
        :rooms="JSON.stringify(rooms)"
        :rooms-loaded="true"
        :messages="JSON.stringify(messages)"
        :messages-loaded="messagesLoaded"
        @send-message="sendMessage($event.detail[0])"
        @fetch-messages="fetchMessages($event.detail[0])"
    />
  </div>
</template>

<script>
import {register} from 'vue-advanced-chat'
import {ref} from "vue";
import axios from "axios";
// import { register } from '../../vue-advanced-chat/dist/vue-advanced-chat.es.js'
register()

export default {
  name: "ChatUI",
  setup() {
    const modelName = ref("");
    const modelOptions = ref([]);
    const loading = ref(false);
    const currentUserId = 'user';
    const rooms = [
      {
        roomId: '1',
        roomName: 'Room 1',
        avatar: 'https://66.media.tumblr.com/avatar_c6a8eae4303e_512.pnj',
        users: [
          {_id: 'system', username: 'System', role: 'system'},
          {_id: 'user', username: 'User', role: 'user'},
          {_id: modelName.value, username: modelName.value, role: 'assistant'}
        ]
      }
    ];
    const messages = ref([]);
    const messagesLoaded = ref(false);

    axios.post('/api/list_models', {})
        .then(function (response) {
          for (const model of response["data"]["models"]) {
            modelOptions.value.push({
              value: model,
              label: model,
              disabled: model === "paddleocr"
            })
            if ((model !== "paddleocr") && (model !== "knowledge")) {
              modelName.value = model
            }
          }
          loading.value = false;
        })
        .catch(function (error) {
          console.log(error);
          loading.value = true;
        });

    function fetchMessages({options = {}}) {
      // TODO: Replace this by retrieving messages from memory according to user_id
      setTimeout(() => {
        if (options.reset) {
          messages.value = addMessages(true)
        } else {
          messages.value = [...addMessages(), ...messages.value]
          messagesLoaded.value = true
        }
        // this.addNewMessage()
      })
    }

    function addMessages(reset) {
      return [{
        _id: reset ? 0 : messages.value.length,
        content: '你好！',
        senderId: 'user',
        username: 'User',
        role: 'user',
        date: '13 November',
        timestamp: '10:20'
      }, {
        _id: reset ? 1 : messages.value.length + 1,
        content: '你好！有什么问题我可以帮助你吗？',
        senderId: modelName.value,
        username: modelName.value,
        role: 'assistant',
        date: '13 November',
        timestamp: '10:20'
      }]
    }

    function isValidJSON(str) {
      try {
        return JSON.parse(str);
      } catch (e) {
        return false;
      }
    }

    async function sendMessage(message) {
      messages.value = [
        ...messages.value,
        {
          _id: messages.value.length,
          content: message.content,
          senderId: currentUserId,
          role: 'user',
          timestamp: new Date().toString().substring(16, 21),
          date: new Date().toDateString()
        }
      ]

      axios.post('/api/rag-chat-completion', {
        model: modelName.value,
        knowledge: "knowledge",
        messages: messages.value,
      })
          .then(function (response) {
            console.log(response)
            messages.value = [
              ...messages.value,
              {
                _id: messages.value.length,
                content: response['data']['text'],
                senderId: modelName.value,
                role: 'assistant',
                timestamp: new Date().toString().substring(16, 21),
                date: new Date().toDateString()
              }
            ]
          })
          .catch(function (error) {
            console.log(error);
          });
    }

    function addNewMessage() {
      setTimeout(() => {
        this.messages = [
          ...this.messages,
          {
            _id: this.messages.length,
            content: 'NEW MESSAGE',
            senderId: '1234',
            timestamp: new Date().toString().substring(16, 21),
            date: new Date().toDateString()
          }
        ]
      }, 2000)
    }

    return {
      modelName,
      modelOptions,
      loading,
      currentUserId,
      rooms,
      messages,
      messagesLoaded,
      fetchMessages,
      addMessages,
      sendMessage,
      addNewMessage
    }
  }
}
</script>

<style lang="scss">
body {
  font-family: 'Quicksand', sans-serif;
}
</style>
