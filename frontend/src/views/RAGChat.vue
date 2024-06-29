<template>
  <a-row :span="24">
    <a-col :span="12">
      <a-card :bordered="true" :style="{ width: '100%' }">
        <ChatUI/>
      </a-card>
    </a-col>
    <a-col :span="12">
      <a-card :bordered="true" :style="{ width: '100%' }">
        <ProxyChat/>
      </a-card>
    </a-col>
  </a-row>
</template>
<script setup>
import ProxyChat from "@/components/RAGChat/ProxyChat.vue";
import ChatUI from "@/components/RAGChat/ChatUI.vue";
</script>