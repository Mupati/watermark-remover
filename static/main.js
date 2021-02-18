const app = new Vue({
  el: "#app",
  data: {
    isProcessing: false,
    isError: false,
    errorMessage: "",
    showDownloadButton: false,
    downloadLink: "",
    currentYear: new Date().getFullYear(),
  },
  methods: {
    processFile() {
      this.isProcessing = true;
      let formData = new FormData();
      formData.append("file", this.$refs.uploadedFile.files[0]);

      axios
        .post("/process", formData, {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        })
        .then((res) => {
          this.isProcessing = false;
          this.showDownloadButton = true;
          this.downloadLink = res.data.processed_file_path;
        })
        .catch((err) => {
          this.isProcessing = false;
          this.isError = true;
          this.errorMesssage = err.response.data.message;
          setTimeout(() => {
            this.isError = false;
            this.errorMessage = "";
          }, 7000);
        });
    },
  },
  delimiters: ["${", "}"],
});
