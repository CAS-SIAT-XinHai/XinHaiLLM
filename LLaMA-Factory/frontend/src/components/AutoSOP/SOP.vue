<!-- eslint-disable vue/multi-word-component-names -->
<template>
  <div class="menu-demo" :style="{'width': '100%', 'text-align': 'center', 'line-height': '0'}">
    <a-menu mode="horizontal" :default-selected-keys="['1']" style="display: inline-block;">
      <a-menu-item key="1" :style="{'font-size': '40px', 'line-height': '1'}">SOP</a-menu-item>
    </a-menu>

    <a-divider/>
    <a-upload draggable action="/"
              :custom-request="customRequest"
    />
    <a-list
        class="list-demo-action-layout"
        :bordered="false"
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
  </div>
</template>

<script>
import {reactive} from 'vue'
import {sopStore} from './store.js'

export default {
  name: "SOP",
  setup() {
    const paginationProps = reactive({
      defaultPageSize: 3,
      total: sopStore.length
    })
    const customRequest = (option) => {
      const {onProgress, onError, onSuccess, fileItem, name} = option
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
            sopStore.push({
              index: sopStore.length,
              title: responseElement.title,
              imageSrc: responseElement.image,
              description: responseElement.description
            })
          }
          paginationProps.total = sopStore.length
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
    const download = () => {
      console.log('点击下载图片')
    };
    return {
      sopStore,
      customRequest,
      download,
      paginationProps
    }
  },
};
</script>

<style scoped>
.list-demo-action-layout .image-area {
  width: 183px;
  height: 119px;
  border-radius: 2px;
  overflow: hidden;
}

.list-demo-item-meta {
  height: 60px;
  overflow: hidden;
}

.list-demo-action-layout .list-demo-item {
  padding: 20px 0;
  border-bottom: 1px solid var(--color-fill-3);
}

.list-demo-action-layout .image-area img {
  width: 100%;
}

.list-demo-action-layout .arco-list-item-action .arco-icon {
  margin: 0 4px;
}
</style>