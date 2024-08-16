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
        @open-file="openFile($event.detail[0])"
        @send-message="sendMessage($event.detail[0])"
        @fetch-messages="fetchMessages($event.detail[0])"
    />
  </div>
</template>

<script>
import {register} from 'vue-advanced-chat'
import {ref} from "vue";
import axios from "axios";

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
              disabled: model === "paddleocr" || model === "knowledge" || model === "storage"
            })
            if (model !== "paddleocr" && model !== "knowledge" && model !== "storage") {
              modelName.value = model
            }
          }
          loading.value = false;
        })
        .catch(function (error) {
          console.log(error);
          loading.value = true;
        });

    function fetchMessages({room}) {
      axios.post('/api/storage/fetch-messages', {
        room: room
      }).then(function (response) {
        messages.value = [...response["data"]["messages"], ...messages.value]
        messagesLoaded.value = true
      }).catch(function (error) {
        console.log(error);
      });
    }

    function isValidJSON(str) {
      try {
        return JSON.parse(str);
      } catch (e) {
        return false;
      }
    }

    function getRoomFromId(room_id) {
      for (const g of rooms) {
        if (g.roomId === room_id) {
          return g
        }
      }
    }

    function blobToBase64(blob) {
      const reader = new FileReader();
      reader.readAsDataURL(blob);
      return new Promise(resolve => {
        reader.onloadend = () => {
          resolve(reader.result);
        };
      });
    };

    function updateFileProgress(messageId, fileUrl, progress) {
      const message = messages.value.find(message => message._id === messageId)

      if (!message || !message.files) return

      message.files.find(file => file.url === fileUrl).progress = progress
      messages.value = [...messages.value]
    }

    function formattedFiles(files) {
      const formattedFiles = []

      files.forEach(file => {
        const messageFile = {
          name: file.name,
          size: file.size,
          type: file.type,
          extension: file.extension || file.type,
          url: file.url || file.localUrl
        }

        if (file.audio) {
          messageFile.audio = true
          messageFile.duration = file.duration
        }

        formattedFiles.push(messageFile)
      })

      return formattedFiles
    }

    function openFile({file}) {
      window.open(file.file.url, '_blank')
    }

    async function uploadFile({file, messageId, roomId}) {
      return new Promise(resolve => {
        let type = file.extension || file.type
        if (type === 'svg' || type === 'pdf') {
          type = file.type
        }

        const xhr = new XMLHttpRequest();
        if (xhr.upload) {
          xhr.upload.onprogress = function (event) {
            let percent;
            if (event.total > 0) {
              // 0 ~ 1
              percent = event.loaded / event.total;
            }
            // onProgress(percent, event);
            updateFileProgress(messageId, file.localUrl, Math.round(percent * 100));
          };
        }
        xhr.onerror = function error(e) {
          resolve(false)
        };
        xhr.onload = function onload() {
          // if (xhr.status < 200 || xhr.status >= 300) {
          //   return onError(xhr.responseText);
          // }
          // onSuccess(xhr.response);
        };
        xhr.onloadend = function onloadend() {
          resolve(true)
        }

        console.log(file)

        const filename = (file.url || file.localUrl) + "." + (file.extension || file.type)

        const formData = new FormData();
        formData.append('file', file.blob, filename);
        xhr.open('post', '/api/upload-file', true);
        xhr.send(formData);
        // return {
        //   abort() {
        //     xhr.abort()
        //   }
        // }
      })
    }

    async function sendMessage({content, roomId, files, replyMessage}) {
      // {
      //   "content": "你好、",
      //   "files": null,
      //   "replyMessage": null,
      //   "usersTag": [],
      //   "roomId": "1"
      // }

      console.log(files);

      const message = {
        _id: messages.value.length,
        indexId: messages.value.length.toString(),
        content: content,
        senderId: currentUserId,
        role: 'user',
        username: 'user',
        timestamp: new Date().toString().substring(16, 21),
        date: new Date().toDateString()
      }

      if (files) {
        message.files = formattedFiles(files);
      }

      messages.value = [
        ...messages.value,
        message
      ]

      if (files) {
        for (let index = 0; index < files.length; index++) {
          await uploadFile({file: files[index], messageId: message._id, roomId})
        }
      }

      try {
        const response = await fetch('/api/chat-completion', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            model: modelName.value,
            messages: messages.value,
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
                        indexId: messages.value.length.toString(),
                        content: content,
                        senderId: modelName.value,
                        role: 'assistant',
                        username: 'assistant',
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
        axios.post('/api/storage/store-messages', {
          room: getRoomFromId(roomId),
          messages: messages.value
        }).then(function (response) {
          console.log(response)
        }).catch(function (error) {
          console.log(error);
        });
      }
    }

    return {
      modelName,
      modelOptions,
      loading,
      currentUserId,
      rooms,
      messages,
      messagesLoaded,
      openFile,
      fetchMessages,
      sendMessage,
    }
  }
}
</script>

<style lang="scss">
body {
  font-family: 'Quicksand', sans-serif;
}
</style>
