/** @type {import('eslint').Linter.Config} */
module.exports = {
  env: {
    es6: true,
    node: true,
    browser: true,
  },
  extends: [
    "eslint:recommended",
  ],
  parser: "@typescript-eslint/parser",
  parserOptions: {
    ecmaVersion: "latest",
    sourceType: "module",
    project: "./tsconfig.json",
  },
  rules: {
    "no-unused-vars": "warn",
    "no-undef": "warn", // 降低为 warning，因为有些全局变量可能在运行时可用
  },
};
