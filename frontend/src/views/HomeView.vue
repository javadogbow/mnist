<template>
  <div class="container">
    <input type="file" ref="preview" @change="setImage"/>
    <button @click="postImage">判定する</button>
    <div class="drawArea">
      <img :src="previewURL"/>
    </div>
    <p class="predict">
      <span class="title">判定結果: </span>
      <span class="value">{{predict}}</span>
    </p>
  </div>
</template>

<script lang="ts">
import axios from "axios";

export default {
  data() {
    return {
      canvas: null,
      context: null,
      isDrag: false,
      file: null,
      previewURL: undefined,
      predict: undefined
    };
  },
  methods: {
    setImage(event) {
      event.preventDefault()
      this.file = event.target.files![0]
      this.previewURL = window.URL.createObjectURL(this.file)
    },
    postImage() {
      axios.postForm(`http://localhost:4000/estimate`, {file: this.file})
          .then((resp) => this.predict = resp.data.predict)
    },
  }
}
</script>

<style scoped>
.container {
  flex-direction: column;
}
button{
  height: 1.5rem;
  width: 4.5rem;
}
img {
  height: 140px;
  width: 140px;
  margin: 20px;
}
.predict {
  font-size: xx-large;
  flex-direction: row;
}
.value {
  font-size: 64px;
  font-weight: bold;
}
</style>
