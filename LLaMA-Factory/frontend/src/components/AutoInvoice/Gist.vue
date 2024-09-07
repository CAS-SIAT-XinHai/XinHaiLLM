<!-- eslint-disable vue/multi-word-component-names -->
<template>
  <a-row :span="24">
    <a-col :span="24">
      <p style="font-size: 30px; color: #333; margin-top: 20px; text-align: left; border-bottom: 1px solid #D9D9D9;">
        基于Invoice的检查要点生成 </p>
      <a-select
          :loading="loading"
          v-model="modelName"
          v-model:options="modelOptions"></a-select>
      <a-textarea placeholder="样例数据" v-model="invoice_content"
                  :style="{'height': '250px',  'float': 'left', 'margin-top': '20px'}">
      </a-textarea>
      <a-textarea placeholder="样例数据" v-model="prompt"
                  :default-value="default_prompt"
                  :style="{'height': '250px',  'float': 'left', 'margin-top': '20px'}">
      </a-textarea>
      <a-button @click="generateGists" :loading="loading" type="primary" shape="round"
                :style="{'float': 'right', 'margin-top': '20px'}" size="large"> 生成审核要点
      </a-button>
    </a-col>
  </a-row>
  <a-row :span="24">
    <a-col :span="12">
      <a-textarea placeholder="样例数据" v-model="gists" auto-size
                  :style="{'float': 'left', 'margin-top': '20px'}">
      </a-textarea>
    </a-col>
    <a-col :span="12">
      <a-list
          v-model:data="gistsStore"
          list-type="picture"
      >
        <template #item="{ item }">
          <a-list-item>
            <a-list-item-meta
                :style="{'backgroundColor': item.color}"
                :title="item.title"
                :description="item.description"
            >
            </a-list-item-meta>
            <a-upload
                v-model:file-list="item.fileList"
                :data="item"
                accept=".rar,.7z,.xlsx,.eml"
                :custom-request="customRequest"
            />
          </a-list-item>
        </template>
      </a-list>
    </a-col>
  </a-row>
</template>

<script>
import {attachmentsStore, gistsStore, invoiceStore} from './store.js'
import {ref} from "vue";
import axios from "axios";

export default {

  // eslint-disable-next-line vue/multi-word-component-names
  name: 'Gist',
  created() {
    document.title = "SOP GISTS";
  },
  // components: {VueMarkdown},
  setup() {
    const modelName = ref("");
    const modelOptions = ref([]);
    const loading = ref(false);
    const default_prompt = "尊敬的项目团队成员，在执行本项目时，我们需要对每个Invoice的内容进行核对，包括金额和附件。\n" +
        "Invoice核心内容概览：\n" +
        "[START OF Invoice]\n" +
        "{content}\n" +
        "[END OF Invoice]\n" +
        "\n" +
        "基于Invoice的内容，生成要点及其细节。" +
        "并用以下格式展示其中的总金额和必要附件：" +
        "{{" +
        "  \"金额\": xxx.xx," +
        "  \"附件\": [" +
        "    \"xxx.txt\"," +
        "    \"yyy.rar\"," +
        "    \"zzz.jpg\"," +
        "  ]" +
        "}}"
    const prompt = ref(default_prompt);
    const invoice_content = ref("");
    const gists = ref("");
    const colors = ["#E3F2FD", "#BBDEFB", "#90CAF9", "#64B5F6"];
    const customRequest = (option) => {
      const {onProgress, onError, onSuccess, fileItem, name, data} = option
      const xhr = new XMLHttpRequest();
      if (xhr.upload) {
        xhr.upload.onprogress = function (event) {
          let percent;
          if (event.total > 0) {
            // 0 ~ 1
            percent = event.loaded / event.total;
          }
          onProgress(percent, event);
        };
      }
      xhr.onerror = function error(e) {
        onError(e);
      };
      xhr.onload = function onload() {
        if (xhr.status < 200 || xhr.status >= 300) {
          return onError(xhr.responseText);
        }
        onSuccess(xhr.response);
      };
      xhr.onloadend = function onloadend() {
        const xhr_ocr = new XMLHttpRequest();
        xhr_ocr.open('post', '/api/parse-file', true);
        xhr_ocr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
        xhr_ocr.responseType = 'json';

        xhr_ocr.onload = function () {
          if (this.status < 200 || this.status >= 300) {
            return onError(this.responseText);
          }

          let response = JSON.parse(this.response);
          for (const responseElement of response["texts"]) {
            attachmentsStore.push({
              index: attachmentsStore.length,
              name: data.title,
              title: responseElement.title,
              imageSrc: responseElement.image,
              description: responseElement.description
            })
          }
        }

        xhr_ocr.send(JSON.stringify({"model": "paddleocr", "filename": fileItem.name}));
        return {
          abort() {
            xhr_ocr.abort()
          }
        }
      }

      const formData = new FormData();
      formData.append(name || 'file', fileItem.file);
      xhr.open('post', '/api/upload-file', true);
      xhr.send(formData);
      return {
        abort() {
          xhr.abort()
        }
      }
    };

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

    function isValidJSON(str) {
      try {
        return JSON.parse(str);
      } catch (e) {
        return false;
      }
    }

    async function generateGists() {
      try {
        loading.value = true;
        gistsStore.splice(0);

        const content = [];
        for (const item of invoiceStore) {
          content.push(item['title'] + item["description"])
        }

        invoice_content.value = content.join("\n\n")

        const response = await fetch('/api/generate-gists', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            model: modelName.value,
            prompt: prompt.value,
            content: invoice_content.value
          })
        });

        const reader = response.body.getReader()
        gists.value = ''

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
                  gists.value += content
                }
                // Do something here
                jsonData = "";
              }
            }
          }
        }
      } catch (error) {
        console.error('Error during random selection:', error);
        loading.value = false;
      } finally {
        loading.value = false;
        let checked = isValidJSON(gists.value)
        if (checked) {
          for (const argument of checked['附件']) {
            gistsStore.push({
              title: argument.split("\n")[0],
              color: colors[gistsStore.length % colors.length],
              description: argument
            })
          }
        }
      }
    }

    return {
      modelName,
      modelOptions,
      default_prompt,
      invoice_content,
      prompt,
      loading,
      gists,
      generateGists,
      customRequest,
      gistsStore,
      attachmentsStore
    }
  },
}

</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
.a-menu {
  width: 100%;
  float: left
}

h3 {
  margin: 40px 0 0;
}

ul {
  list-style-type: none;
  padding: 0;
}

li {
  display: inline-block;
  margin: 0 10px;
}

a {
  color: #42b983;
}

.grid-demo .arco-col {
  height: 48px;
  line-height: 48px;
  text-align: center;
}
</style>
