<!-- eslint-disable vue/multi-word-component-names -->
<template>
  <a-row>
    <a-col :span="24">
      <p style="font-size: 30px; color: #333; margin-top: 20px; text-align: left; border-bottom: 1px solid #D9D9D9;">
        基于检查要点的审核 </p>
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
      <a-button @click="auditGists" :loading="loading" type="primary" shape="round"
                :style="{'float': 'right', 'margin-top': '20px'}" size="large"> 执行审核
      </a-button>
    </a-col>
  </a-row>
  <a-row>
    <a-col :span="24">
      <a-list
          :bordered="false"
          v-model:data="auditedGists"
          :style="{'margin-top': '20px'}"
      >
        <template #item="{ item }">
          <a-list-item class="list-demo-item" style="{'padding': 0}">
            <a-list-item-meta
                :style="{'backgroundColor': item.color}"
                :title="item.title"
                :description="item.description"
            >
              <template #avatar>
                <a-avatar shape="square">
                  <a-image
                      width="200"
                      :src="item.avatar"
                      :preview-props="{
      actionsLayout: ['rotateRight', 'zoomIn', 'zoomOut'],
    }"
                  >
                    <template #preview-actions>
                      <a-image-preview-action name="下载" @click="download">
                        <icon-download/>
                      </a-image-preview-action>
                    </template>
                  </a-image>
                </a-avatar>
              </template>
            </a-list-item-meta>
            <a-textarea placeholder="样例数据" v-model="item.audit" auto-size
                        :style="{'float': 'left', 'margin-top': '10px', 'margin-bottom': '20px'}">
            </a-textarea>
          </a-list-item>
        </template>
      </a-list>
    </a-col>
  </a-row>
</template>

<script>
import {gistsStore, invoiceStore} from './store.js'
import {ref} from "vue";
import axios from "axios";

export default {

  // eslint-disable-next-line vue/multi-word-component-names
  name: 'Gist',
  created() {
    document.title = "SOP GISTS";
  },
  setup() {
    const modelName = ref("");
    const modelOptions = ref([]);
    const loading = ref(false);
    const default_prompt = "结合SOP中的检查要点，针对提交的文件进行审核，并指出风险点。\n" +
        "\n" +
        "[START OF AUDIT RULE]\n" +
        "{gist}\n" +
        "[END OF AUDIT RULE]\n" +
        "\n" +
        "[START OF INVOICE]\n" +
        "{invoice_content}\n" +
        "[END OF INVOICE]\n"
    const prompt = ref(default_prompt);
    const auditedGists = ref([]);
    const invoice_content = ref("");

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

    async function auditGists() {
      try {
        loading.value = false;
        invoice_content.value = invoiceStore[0].description
        auditedGists.value = []
        for (const g of gistsStore) {
          auditedGists.value.push({
            title: g.title,
            color: g.color,
            description: g.description,
            audit: ""
          })
        }

        const response = await fetch('/api/audit-gists', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            model: modelName.value,
            prompt: prompt.value,
            gists: gistsStore,
            invoice: invoice_content.value
          })
        });

        const reader = response.body.getReader()
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
                  auditedGists.value[checked["gist_id"]].audit += content
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
      }
    }

    return {
      modelName,
      modelOptions,
      default_prompt,
      prompt,
      invoice_content,
      loading,
      auditGists,
      auditedGists
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
