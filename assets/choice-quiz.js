document.querySelectorAll('[data-choice-quiz]').forEach(form => {
  const feedback = form.querySelector('[data-feedback]');
  form.addEventListener('change', () => { feedback.textContent = ''; });
  form.addEventListener('submit', event => {
    event.preventDefault();
    const answer = new FormData(form).get('approach');
    feedback.textContent = answer === 'examples'
      ? '对，这里由算法参考例子形成判断方式。人仍要准备数据和检查效果；接下来试着用自己的话解释邮件例子。'
      : '这描述的是人直接指定判断规则。再想想：哪一种做法会让例子参与形成判断方式？';
  });
});
