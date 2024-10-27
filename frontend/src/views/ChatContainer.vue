<template>
  <a-card :bordered="true" :style="{ width: '100%' }">
    <ChatUI/>
  </a-card>
</template>
<script setup>
import ChatUI from "@/components/ChatContainer/ChatUI.vue";
</script>