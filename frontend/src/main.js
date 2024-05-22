import { createApp } from 'vue'
import ArcoVue from '@arco-design/web-vue';
import App from './App.vue';
import '@arco-design/web-vue/dist/arco.css';
import Axios from 'axios'
import ArcoVueIcon from '@arco-design/web-vue/es/icon';
import router from './router'

import {Message} from '@arco-design/web-vue'


const app = createApp(App);

Message._context = app._context;
app.config.globalProperties.$message = Message;
app.use(ArcoVue)
    .use(router)
    .use(ArcoVueIcon)
    .mount('#app');
app.config.globalProperties.$http = Axios;
