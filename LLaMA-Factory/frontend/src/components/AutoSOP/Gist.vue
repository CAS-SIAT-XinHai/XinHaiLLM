<!-- eslint-disable vue/multi-word-component-names -->
<template>
  <a-row :span="24">
    <a-col :span="24">
      <p style="font-size: 30px; color: #333; margin-top: 20px; text-align: left; border-bottom: 1px solid #D9D9D9;">
        基于SOP的检查要点生成 </p>
      <a-select
          :loading="loading"
          v-model="modelName"
          v-model:options="modelOptions"></a-select>
      <a-textarea placeholder="样例数据" v-model="sop_content"
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
      <a-textarea placeholder="样例数据" v-model="gists"
                  :style="{'height': '500px', 'float': 'left', 'margin-top': '20px'}">
      </a-textarea>
    </a-col>
    <a-col :span="12">
      <a-list
          v-model:data="gistsStore"
          :style="{'margin-top': '20px'}"
      >
        <template #item="{ item }">
          <a-list-item>
            <a-list-item-meta
                :style="{'backgroundColor': item.color}"
                :title="item.title"
                :description="item.description"
            >
            </a-list-item-meta>
          </a-list-item>
        </template>
      </a-list>
    </a-col>
  </a-row>
</template>

<script>
import {gistsStore, sopStore} from './store.js'
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
    const default_prompt = "尊敬的项目团队成员，在执行本项目时，我们依赖一系列与标准作业程序（SOP）相关的文件来确保流程的一致性和效率。\n" +
        "SOP核心内容概览：\n" +
        "[START OF SOP]\n" +
        "{content}\n" +
        "[END OF SOP]\n" +
        "\n" +
        "基于SOP的内容，生成要点及其细节。不要重复SOP的内容。\n"
        // "生成格式为如下json:\n" +
        // "```\n" +
        // "[\n" +
        // "  {{\n" +
        // "    \"要点\": xxx,\n" +
        // "    \"内容\": []\n" +
        // "  }}\n" +
        // "]\n" +
        // "```"
    const prompt = ref(default_prompt);
    const sop_content = ref("");
    const gists = ref("");
    const colors = ["#E3F2FD", "#BBDEFB", "#90CAF9", "#64B5F6"];

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

        const sop = [];
        for (const item of sopStore) {
          sop.push(item['title'] + item["description"])
        }

        sop_content.value = sop.join("\n\n")

        const response = await fetch('/api/generate-gists', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            model: modelName.value,
            prompt: prompt.value,
            content: sop_content.value
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
        for (const argument of gists.value.split(/\d+\./)) {
          gistsStore.push({
            title: argument.split("\n")[0],
            color: colors[gistsStore.length % colors.length],
            description: argument
          })
        }
      }
    }

    return {
      modelName,
      modelOptions,
      default_prompt,
      sop_content,
      prompt,
      loading,
      gists,
      generateGists,
      gistsStore
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
