<!-- eslint-disable vue/multi-word-component-names -->
<template>
  <div class="menu-demo" :style="{'width': '100%', 'text-align': 'center', 'line-height': '0'}">
    <a-menu mode="horizontal" :default-selected-keys="['1']" style="display: inline-block;">
      <a-menu-item key="1" :style="{'font-size': '40px', 'line-height': '1'}">Invoice</a-menu-item>
    </a-menu>

    <a-divider/>
    <a-upload
        list-type="picture"
        v-model:file-list="fileList"
        :custom-request="customRequest"
    />
    <a-list
        class="list-demo-action-layout"
        :bordered="false"
        v-model:data="invoiceStore"
        :pagination-props="paginationProps"
    >
      <template #item="{ item }">
        <a-list-item class="list-demo-item" action-layout="vertical">
<!--          <template #actions>-->
<!--            <span><icon-heart/>83</span>-->
<!--            <span><icon-star/>{{ item.index }}</span>-->
<!--            <span><icon-message/>Reply</span>-->
<!--          </template>-->
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
        </a-list-item>
      </template>
    </a-list>
  </div>
</template>

<script>
import {reactive, ref} from 'vue'
import {invoiceStore} from './store.js'

export default {
  setup() {
    const fileList = ref([]);
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
        xhr_ocr.open('post', '/api/ocr-image', true);
        xhr_ocr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
        xhr_ocr.responseType = 'json';

        xhr_ocr.onload = function () {
          if (this.status < 200 || this.status >= 300) {
            return onError(this.responseText);
          }

          let response = JSON.parse(this.response);

          invoiceStore.push({
            index: fileList.value.length,
            avatar: fileItem.url,
            title: fileItem.name,
            description: response["description"],
            imageSrc: response["image"],
          })
        }

        xhr_ocr.send(JSON.stringify({"model": "paddleocr", "image": fileItem.name}));
        return {
          abort() {
            xhr_ocr.abort()
          }
        }
      }

      const formData = new FormData();
      formData.append(name || 'file', fileItem.file);
      xhr.open('post', '/api/upload-image', true);
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
      fileList,
      invoiceStore,
      customRequest,
      download,
      paginationProps: reactive({
        defaultPageSize: 3,
        total: invoiceStore.length
      })
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

.list-demo-action-layout .list-demo-item {
  padding: 20px 0;
  border-bottom: 1px solid var(--color-fill-3);
}

.list-demo-item-meta {
  height: 60px;
  overflow: hidden;
}

.list-demo-action-layout .image-area img {
  width: 100%;
}

.list-demo-action-layout .arco-list-item-action .arco-icon {
  margin: 0 4px;
}
</style>