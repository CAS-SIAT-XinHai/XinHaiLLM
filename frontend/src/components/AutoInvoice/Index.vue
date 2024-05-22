<!-- eslint-disable vue/multi-word-component-names -->
<template>
  <a-row>
    <a-col :span="12">
      <!--      <a-select v-model="from" :options="fromOptions"-->
      <!--                :style="{width:'200px', 'margin-left': '30px', 'borderRadius': '10px', 'backgroundColor': '#FFFFFF', 'border': '1px solid #D9D9D9', 'float': 'left'}"></a-select>-->
      <a-list
          v-model:data="sopStore"
          :pagination-props="paginationProps"
      >
        <template #item="{ item }">
          <a-list-item>
            <template #extra>
              <div className="image-area">
                <a-image
                    width="200"
                    :src="`data:image/png;base64,${item.imageSrc}`"
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
              </div>
            </template>
            <a-list-item-meta class="list-demo-item-meta"
                              :title="item.title"
                              :description="item.description"
            >
            </a-list-item-meta>
          </a-list-item>
        </template>
      </a-list>
      <!--      <a-textarea placeholder="样例数据" v-model="sopStore"-->
      <!--                  :style="{'height': '250px',  'float': 'left', 'margin-left': '30px', 'margin-top': '20px'}">-->
      <!--      </a-textarea>-->
      <!--      <a-button @click="random" :loading="loading" type="primary" shape="round"-->
      <!--                :style="{'float': 'right', 'margin-top': '20px'}" size="large">Random Select-->
      <!--      </a-button>-->
    </a-col>
    <a-col :span="12">
      <a-select v-model="from" :options="fromOptions"
                :style="{width:'200px', 'margin-left': '30px', 'borderRadius': '10px', 'backgroundColor': '#FFFFFF', 'border': '1px solid #D9D9D9', 'float': 'left'}"></a-select>
      <a-textarea placeholder="样例数据" v-model="output_1"
                  :style="{'height': '250px',  'float': 'left', 'margin-left': '30px', 'margin-top': '20px'}">
      </a-textarea>
      <a-button @click="random" :loading="loading" type="primary" shape="round"
                :style="{'float': 'right', 'margin-top': '20px'}" size="large">Random Select
      </a-button>
    </a-col>
    <a-col :span="1">
    </a-col>
  </a-row>
  <a-row class="grid-demo" :flex="true">
    <a-col :span="23" :style="{ 'margin-left': '30px', 'position': 'relative' }">
      <p style="font-size: 16px; color: #333; margin-top: 320px; text-align: left; margin-bottom: 60px; border-bottom: 1px solid #D9D9D9;">
        Retrieved Document Chunks (4 items)</p>
    </a-col>
  </a-row>
  <a-row class="grid-demo" :flex="true">
    <a-col :span="5" :style="{ 'margin-left': '30px' }">
      <a-textarea
          :autosize="true"
          placeholder="Chunk1"
          v-model="chunks[0].content"
          :style="{'width': '100%', 'height':'180px','margin-top': '350px', 'background-color': '#FFFFFF','borderRadius': '10px', 'backgroundColor': '#FFFFFF', 'border': '1px solid #D9D9D9',}"
      ></a-textarea>
    </a-col>
    <a-col :span="5" :style="{ 'margin-left': '63px' }">
      <a-textarea
          :autosize="true"
          placeholder="Chunk2"
          v-model="chunks[1].content"
          :style="{'width': '100%', 'height':'180px','margin-top': '350px', 'background-color': '#FFFFFF', 'borderRadius': '10px', 'backgroundColor': '#FFFFFF', 'border': '1px solid #D9D9D9',}"
      ></a-textarea>
    </a-col>
    <a-col :span="1"></a-col>
    <a-col :span="5">
      <a-textarea
          :autosize="true"
          placeholder="Chunk3"
          v-model="chunks[2].content"
          :style="{'width': '100%', 'height':'180px','margin-top': '350px', 'background-color': '#FFFFFF','borderRadius': '10px', 'backgroundColor': '#FFFFFF', 'border': '1px solid #D9D9D9',}"
      ></a-textarea>
    </a-col>
    <a-col :span="5" :style="{ 'margin-left': '63px' }">
      <a-textarea
          :autosize="true"
          placeholder="Chunk4"
          v-model="chunks[3].content"
          :style="{'width': '100%','height':'180px','margin-top': '350px', 'background-color': '#FFFFFF', 'borderRadius': '10px', 'backgroundColor': '#FFFFFF', 'border': '1px solid #D9D9D9',}"
      ></a-textarea>
    </a-col>
  </a-row>
</template>

<script>
import {sopStore} from './store.js'
import {reactive} from "vue";

export default {

  // eslint-disable-next-line vue/multi-word-component-names
  name: 'Index',
  created() {
    document.title = "RAG";
  },
  setup() {
    const download = () => {
      console.log('点击下载图片')
    };
    return {
      download,
      paginationProps: reactive({
        defaultPageSize: 3,
        total: sopStore.length
      })
    }
  },
  data() {
    return {
      sopStore,
      fromOptions: [
        {value: 'PsyQA_train', label: 'PsyQA_train'},
      ],
      modelOptions: [
        {value: 'zhipuai', label: 'zhipuai'},
        {value: 'gpt_3.5', label: 'gpt_3.5'},
        {value: 'chatglm3-6b', label: 'chatglm3-6b'}
      ],
      from: 'PsyQA_train',
      model: 'zhipuai',
      input: '',
      output: '',
      loading: false,
      chunks: [
        {content: ''},
        {content: ''},
        {content: ''},
        {content: ''},
      ],
    }
  },
  methods: {
    async random() {
      try {
        this.loading = true;
        if (this.socket) {
          this.socket.close();
        }
        const response = await fetch('/api/random', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            selectedSource: this.from
          })
        });

        const responseData = await response.json();

        this.output_1 = responseData.data;
      } catch (error) {
        console.error('Error during random selection:', error);
      } finally {
        this.loading = false;
      }
    },

    async Query() {
      try {
        this.loading = true;

        const selectedTextData = this.output_1;

        const response = await fetch('/api/rag_query', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            selectedModel: this.model,
            selectedData: selectedTextData
          })
        });

        const responseData = await response.json();

        this.chunks[0].content = responseData.chunks[0]
        this.chunks[1].content = responseData.chunks[1]
        this.chunks[2].content = responseData.chunks[2]
        this.chunks[3].content = responseData.chunks[3]
        const rewrittenResult = responseData.rewritten;

        this.output_2 = rewrittenResult;

      } catch (error) {
        console.error('Error during query:', error);
      } finally {
        this.loading = false;
      }
    },

    async Eval() {
      try {
        this.loading = true;

        const randomResult = this.output_1;
        const queryResult = this.output_2;
        const selectedModel = this.model;

        const response = await fetch('/api/evaluate', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            selectedModel: selectedModel,
            randomResult: randomResult,
            queryResult: queryResult
          })
        });

        const responseData = await response.json();

        this.output_3 = responseData.evaluationResult;

      } catch (error) {
        console.error('Error during evaluation:', error);
      } finally {
        this.loading = false;
      }
    },
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
