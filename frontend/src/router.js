import {createMemoryHistory, createRouter} from 'vue-router'
import AutoSOP from "@/views/AutoSOP.vue";
import AutoInvoice from "@/views/AutoInvoice.vue";

const routes = [
    {path: '/autosop', component: AutoSOP},
    {path: '/autoinvoice', component: AutoInvoice},
]

const router = createRouter({
    history: createMemoryHistory(),
    routes,
})

export default router