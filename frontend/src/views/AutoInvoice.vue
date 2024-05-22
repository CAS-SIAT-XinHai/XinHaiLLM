<template>
  <a-row :span="24">
    <a-card :bordered="true" :style="{ width: '100%' }">
      <Invoice/>
    </a-card>
  </a-row>
  <a-row>
    <a-col :span="24">
      <a-card :bordered="true" :style="{ width: '100%' }">
        <div class="menu-demo" :style="{'width': '100%', 'text-align': 'center', 'line-height': '0'}">
          <a-menu mode="horizontal" :default-selected-keys="['1']" style="display: inline-block;">
            <a-menu-item key="1" :style="{'font-size': '40px', 'line-height': '1'}">AutoInvoice</a-menu-item>
          </a-menu>
        </div>
        <a-divider/>
        <a-col :span="24">
          <Gist/>
        </a-col>
        <a-col :span="24">
          <Auditing/>
        </a-col>
      </a-card>
    </a-col>
  </a-row>
</template>
<script setup>
import Gist from "@/components/AutoInvoice/Gist.vue";
import Auditing from "@/components/AutoInvoice/Auditing.vue";
import Invoice from "@/components/AutoInvoice/Invoice.vue";
</script>