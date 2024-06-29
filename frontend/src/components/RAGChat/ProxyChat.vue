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
  name: "ProxyChat",
  setup() {
    const modelName = ref("");
    const modelOptions = ref([]);
    const loading = ref(false);
    const currentUserId = '1234';
    const rooms = [
      {
        roomId: '1',
        roomName: 'Room 1',
        avatar: 'https://66.media.tumblr.com/avatar_c6a8eae4303e_512.pnj',
        users: [
          {_id: '1234', username: 'John Doe'},
          {_id: modelName.value, username: modelName.value}
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
            if (model !== "paddleocr") {
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
      const temp_messages = []

      for (let i = 0; i < 30; i++) {
        temp_messages.push({
          _id: reset ? i : messages.value.length + i,
          content: `${reset ? '' : 'paginated'} message ${i + 1}`,
          senderId: '4321',
          username: 'John Doe',
          date: '13 November',
          timestamp: '10:20'
        })
      }

      return temp_messages
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
          timestamp: new Date().toString().substring(16, 21),
          date: new Date().toDateString()
        }
      ]

      try {
        const response = await fetch('/api/chat-completion', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            model: modelName.value,
            content: message.content,
          })
        });

        const reader = response.body.getReader()

        let m_len = messages.value.length
        let done = false
        while (!done) {
          const {done, value} = await reader.read()
          if (done) break

          const chunkStr = new TextDecoder('utf-8').decode(value)

          let jsonData = "";
          // Loop through the chars until we get a valid JSON object
          for (var x = 0; x < chunkStr.length; x++) {
            let last_char = chunkStr.charAt(x)
            jsonData += last_char;
            if (last_char === "}") {
              let checked = isValidJSON(jsonData)
              if (checked) {
                const {
                  choices: [
                    {
                      delta: {content},
                    },
                  ],
                } = checked
                if (content) {
                  if (m_len === messages.value.length) {
                    messages.value = [
                      ...messages.value,
                      {
                        _id: messages.value.length,
                        content: content,
                        senderId: modelName.value,
                        timestamp: new Date().toString().substring(16, 21),
                        date: new Date().toDateString()
                      }
                    ]
                  } else {
                    messages.value[messages.value.length - 1].content += content
                  }
                }
                // Do something here
                jsonData = "";
              }
            }
          }
        }
      } catch (error) {
        console.error('Error during random selection:', error);
      } finally {
        // let checked = isValidJSON(gists.value)
        // if (checked) {
        // }
      }
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
