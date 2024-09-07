import {createMemoryHistory, createRouter} from 'vue-router'
import AutoSOP from "@/views/AutoSOP.vue";
import AutoInvoice from "@/views/AutoInvoice.vue";
import ChatContainer from "@/views/ChatContainer.vue";
import RAGChat from "@/views/RAGChat.vue";
import SimulationContainer from "@/views/SimulationContainer.vue";

const routes = [
    {path: '/chat', component: ChatContainer},
    {path: '/simulation', component: SimulationContainer},
    {path: '/ragchat', component: RAGChat},
    {path: '/autosop', component: AutoSOP},
    {path: '/autoinvoice', component: AutoInvoice},
]

const router = createRouter({
    history: createMemoryHistory(),
    routes,
})

export default router