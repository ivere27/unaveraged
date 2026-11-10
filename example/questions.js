// Presets describe the selected receipt. Only known annotation answers can
// produce a correctness badge; arbitrary user questions remain ungraded.
const clean = value => String(value ?? '').normalize('NFC').replace(/\s+/gu, ' ').trim();
const numberText = value => (typeof value === 'number' && Number.isFinite(value))
  || (typeof value === 'string' && /^\d+$/.test(value)) ? String(value) : undefined;

export function questionPresets(sample, model) {
  if (!sample || !['digit', 'vqa'].includes(model)) return [];
  const presets = [], seen = new Set();
  const add = (id, label, family, question, answer) => {
    question = clean(question);
    if (!question) return;
    if (seen.has(question)) {
      const original = presets.find(item => item.question === question);
      // Keep the published default/order, with the concise generated label.
      if (original.id.startsWith('source-') && !id.startsWith('source-')) {
        original.label = label;
        original.family = family;
      }
      return;
    }
    seen.add(question);
    const expected = answer == null ? undefined : clean(answer);
    presets.push({ id, label, family, question, ...(expected ? { answer: expected } : {}) });
  };
  for (const [index, item] of (sample.questions || []).entries()) {
    add(`source-${index}`, item.label || item.question, item.family || '기본', item.question, item.answer);
  }
  const receipt = sample.receipt || {};
  const phone = /^\d+$/.test(receipt.phone_digits || '') ? receipt.phone_digits : undefined;
  const street = numberText(receipt.street_no);
  for (const side of ['back', 'front']) {
    for (let ordinal = 1; ordinal <= Math.min(3, phone?.length ?? 3); ordinal++) {
      const direction = side === 'front' ? '앞' : '뒤';
      add(`phone-${side}-${ordinal}`, `전화번호 · ${direction}에서 ${ordinal}번째`, '전화번호',
        `가게 전화번호의 ${direction}에서 ${ordinal}번째 숫자는 무엇입니까?`,
        phone?.[side === 'front' ? ordinal - 1 : phone.length - ordinal]);
    }
  }
  add('street-number', '도로명 뒤 숫자', '주소', '영수증의 가게 주소에서 도로명 뒤 숫자는 무엇입니까?', street);
  if (model === 'digit') return presets;

  add('phone-number', '전체 전화번호', '전화번호', '가게 전화번호는 무엇입니까?', phone);
  if (clean(receipt.store)) add('store-name', '상호명', '가게', '가게 이름은 무엇입니까?', receipt.store);
  if (clean(receipt.address)) add('store-address', '전체 주소', '주소', '가게의 전체 주소는 무엇입니까?', receipt.address);
  const items = Array.isArray(receipt.items) ? receipt.items : [];
  for (const [index, item] of items.slice(0, 3).entries()) {
    const n = index + 1;
    if (clean(item.name)) add(`item-${n}-name`, `${n}번째 품목명`, '품목', `${n}번째 품목명은 무엇입니까?`, item.name);
    if (numberText(item.quantity) !== undefined) add(`item-${n}-quantity`, `${n}번째 품목 수량`, '품목',
      `${n}번째 품목의 개수는 몇 개입니까?`, item.quantity);
    if (numberText(item.unit_price) !== undefined) add(`item-${n}-unit-price`, `${n}번째 품목 단가`, '품목',
      `${n}번째 품목의 단가는 얼마입니까?`, item.unit_price);
    if (numberText(item.total) !== undefined) add(`item-${n}-total`, `${n}번째 품목 금액`, '품목',
      `${n}번째 품목의 총합은 얼마입니까?`, item.total);
    if (clean(item.name) && items.filter(other => clean(other.name) === clean(item.name)).length === 1
        && numberText(item.quantity) !== undefined) {
      add(`lookup-${n}-quantity`, `${clean(item.name)} · 수량`, '상품별 조회', `${clean(item.name)} 개수는 몇 개입니까?`, item.quantity);
    }
  }
  if (items.length && items.every(item => numberText(item.total) !== undefined)) {
    add('items-total', '품목 금액 합계', '합계', '품목 총합을 모두 더하면 얼마입니까?',
      items.reduce((sum, item) => sum + Number(item.total), 0));
  }
  if (items.length && items.every(item => numberText(item.quantity) !== undefined)) {
    add('items-quantity', '품목 수량 합계', '합계', '품목 개수를 모두 더하면 몇 개입니까?',
      items.reduce((sum, item) => sum + Number(item.quantity), 0));
  }
  return presets;
}

export function expectedAnswer(sample, model, question) {
  const normalized = clean(question);
  return questionPresets(sample, model).find(item => item.question === normalized)?.answer;
}
