const state = {
  user: null,
  csrfToken: null,
  genres: [],
  favoriteMovies: new Set(),
  authMode: "login",
  toastTimer: null,
  guestPage: 1,
  recommendationPage: 1,
  profileMoviePage: 1,
  profileMovieQuery: "",
  adminUserPage: 1,
  adminMoviePage: 1,
  adminMovieQuery: "",
  profileSearchTimer: null
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

async function api(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (options.body && !(options.body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
  }
  if (state.csrfToken && ["POST", "PUT", "PATCH", "DELETE"].includes((options.method || "GET").toUpperCase())) {
    headers["X-CSRF-Token"] = state.csrfToken;
  }
  const response = await fetch(path, { ...options, headers, credentials: "same-origin" });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || "Có lỗi xảy ra, vui lòng thử lại.");
  return payload;
}

function toast(message) {
  const element = $("#toast");
  element.textContent = message;
  element.classList.add("visible");
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => element.classList.remove("visible"), 3200);
}

function escapeText(value) {
  return String(value ?? "");
}

function createMovieCard(movie) {
  const article = document.createElement("article");
  article.className = "movie-card";
  article.tabIndex = 0;
  article.setAttribute("role", "button");
  article.setAttribute("aria-label", `Chi tiết phim ${movie.title}`);
  article.addEventListener("click", () => openMovieDetails(movie));
  article.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      openMovieDetails(movie);
    }
  });
  const title = document.createElement("h3");
  title.textContent = escapeText(movie.title);
  const info = document.createElement("div");
  info.className = "movie-info";
  const genres = document.createElement("p");
  genres.className = "movie-genres";
  genres.textContent = Array.isArray(movie.genres) ? movie.genres.slice(0, 3).join(" · ") : "";
  const overview = document.createElement("p");
  overview.className = "movie-description";
  overview.textContent = movie.overview || "Chưa có mô tả cho phim này.";
  const ratings = document.createElement("div");
  ratings.className = "movie-card-ratings";
  const score = document.createElement("span");
  score.className = "movie-score";
  score.textContent = `★ ${Number(movie.rating_average || 0).toFixed(2)} / 5`;
  const count = document.createElement("span");
  count.className = "movie-count";
  count.textContent = `${Number(movie.rating_count || 0).toLocaleString("vi-VN")} lượt đánh giá`;
  ratings.append(score, count);
  info.append(title, genres, overview);
  article.append(info, ratings);
  return article;
}

function renderMovies(element, movies) {
  element.replaceChildren();
  if (!movies?.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "Chưa có phim phù hợp.";
    element.append(empty);
    return;
  }
  movies.forEach((movie) => element.append(createMovieCard(movie)));
}

function renderPagination(container, currentPage, totalPages, onPageChange) {
  container.replaceChildren();
  if (totalPages <= 1) return;
  const previous = document.createElement("button");
  previous.className = "page-button";
  previous.textContent = "← Trước";
  previous.disabled = currentPage <= 1;
  previous.addEventListener("click", () => onPageChange(currentPage - 1));
  const label = document.createElement("span");
  label.className = "page-label";
  label.textContent = `Trang ${currentPage} / ${totalPages}`;
  const next = document.createElement("button");
  next.className = "page-button";
  next.textContent = "Tiếp →";
  next.disabled = currentPage >= totalPages;
  next.addEventListener("click", () => onPageChange(currentPage + 1));
  container.append(previous, label, next);
}

async function loadGuestContent(page = state.guestPage) {
  state.guestPage = page;
  const genre = $("#genre-filter").value;
  const data = await api(`/api/guest?page=${page}&page_size=10&genre=${encodeURIComponent(genre)}`);
  const moviePage = data.movies;
  renderMovies($("#popular-movies"), moviePage.items);
  const genreFilter = $("#genre-filter");
  const selectedGenre = genreFilter.value;
  genreFilter.replaceChildren(new Option("Tất cả thể loại", ""));
  data.genres.forEach((genreName) => genreFilter.add(new Option(genreName, genreName)));
  genreFilter.value = selectedGenre;
  renderPagination($("#popular-pagination"), moviePage.page, moviePage.total_pages, loadGuestContent);
}

function updateRoleUI() {
  const loggedIn = Boolean(state.user);
  $("#account-label").textContent = loggedIn ? `Xin chào, ${state.user.display_name}` : "Khách";
  $("#login-open").classList.toggle("hidden", loggedIn);
  $("#register-open").classList.toggle("hidden", loggedIn);
  $("#logout-button").classList.toggle("hidden", !loggedIn);
  $("#personalized").classList.toggle("hidden", !loggedIn || state.user.role === "admin");
  $("#admin-panel").classList.toggle("hidden", !loggedIn || state.user.role !== "admin");
  if (loggedIn && state.user.role === "admin") loadAdminUsers();
  if (loggedIn && state.user.role !== "admin") loadRecommendations();
}

async function loadSession() {
  const data = await api("/api/session");
  state.user = data.user;
  state.csrfToken = data.csrf_token;
  updateRoleUI();
}

async function loadRecommendations() {
  try {
    const data = await api(`/api/recommendations?page=${state.recommendationPage}&page_size=10`);
    const hasPreferences =
      state.user?.favorite_genres?.length || state.user?.favorite_movie_ids?.length;
    const source = data.recommendation_source;
    $("#personalized-note").textContent = source === "hybrid_gnn_llm"
      ? "Gợi ý bằng embedding Hybrid GNN + SentenceTransformer theo sở thích bạn đã chọn."
      : source === "genre_filter"
        ? "Mô hình chưa có kết quả phù hợp; danh sách tạm thời được lọc theo thể loại bạn đã chọn."
        : hasPreferences
          ? "Chưa tìm được vector phù hợp với sở thích; hiện đang hiển thị phim rating cao."
          : "Bạn chưa nhập sở thích — danh sách hiện là phim rating cao. Cập nhật hồ sơ để bật gợi ý cá nhân hóa.";
    renderMovies($("#recommendation-list"), data.items);
    renderPagination(
      $("#recommendation-pagination"),
      data.page,
      data.total_pages,
      (page) => {
        state.recommendationPage = page;
        loadRecommendations();
      }
    );
  } catch (error) {
    toast(error.message);
  }
}

function openMovieDetails(movie) {
  $("#movie-detail-title").textContent = movie.title || "";
  $("#movie-detail-genres").textContent = (movie.genres || []).join(" · ") || "Chưa phân loại";
  $("#movie-detail-overview").textContent = movie.overview || "Chưa có mô tả cho phim này.";
  const movielensRating =
    `${Number(movie.rating_average || 0).toFixed(2)} / 5 từ ` +
    `${Number(movie.rating_count || 0).toLocaleString("vi-VN")} lượt MovieLens`;
  const tmdbRating = Number(movie.vote_average || 0);
  $("#movie-detail-ratings").textContent = tmdbRating
    ? `${movielensRating} · TMDB ${tmdbRating.toFixed(1)} / 10`
    : movielensRating;
  $("#movie-dialog").showModal();
}

function setAuthMode(mode) {
  state.authMode = mode;
  const register = mode === "register";
  $("#auth-title").textContent = register ? "Tạo tài khoản" : "Đăng nhập";
  $("#auth-subtitle").textContent = register
    ? "Tạo hồ sơ để bắt đầu khám phá phim hợp gu."
    : "Đăng nhập để khám phá phim hợp gu.";
  $("#name-field").classList.toggle("hidden", !register);
  $("#name-field").querySelector("input").required = register;
  $("#auth-form input[name=password]").autocomplete = register ? "new-password" : "current-password";
  $("#auth-submit").textContent = register ? "Tạo tài khoản" : "Đăng nhập";
  $("#auth-switch").textContent = register
    ? "Đã có tài khoản? Đăng nhập"
    : "Chưa có tài khoản? Tạo tài khoản";
  $("#auth-error").textContent = "";
}

async function submitAuth(event) {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  const payload = {
    email: form.get("email"),
    password: form.get("password")
  };
  if (state.authMode === "register") payload.display_name = form.get("display_name");
  try {
    const data = await api(`/api/auth/${state.authMode}`, {
      method: "POST",
      body: JSON.stringify(payload)
    });
    state.user = data.user;
    state.csrfToken = data.csrf_token;
    state.recommendationPage = 1;
    $("#auth-dialog").close();
    event.currentTarget.reset();
    updateRoleUI();
    toast(state.authMode === "register" ? "Tài khoản đã được tạo." : "Đăng nhập thành công.");
    if (state.user.role === "user") openProfile();
  } catch (error) {
    $("#auth-error").textContent = error.message;
  }
}

async function logout() {
  try {
    await api("/api/auth/logout", { method: "POST", body: "{}" });
    state.user = null;
    state.csrfToken = null;
    state.recommendationPage = 1;
    updateRoleUI();
    toast("Bạn đã đăng xuất.");
  } catch (error) {
    toast(error.message);
  }
}

async function openProfile() {
  if (!state.user) {
    $("#auth-dialog").showModal();
    return;
  }
  try {
    const data = await api("/api/profile");
    state.user = data.user;
    state.genres = data.genres;
    state.favoriteMovies = new Set(data.user.favorite_movie_ids);
    $("#profile-form input[name=display_name]").value = data.user.display_name;
    const genres = $("#genre-choices");
    genres.replaceChildren();
    data.genres.forEach((genre) => {
      const label = document.createElement("label");
      label.className = "choice-chip";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.name = "favorite_genres";
      checkbox.value = genre;
      checkbox.checked = data.user.favorite_genres.includes(genre);
      const text = document.createElement("span");
      text.textContent = genre;
      label.append(checkbox, text);
      genres.append(label);
    });
    state.profileMoviePage = 1;
    state.profileMovieQuery = "";
    $("#favorite-search").value = "";
    renderMovieChoices(data.movies.items);
    renderPagination(
      $("#profile-movie-pagination"),
      data.movies.page,
      data.movies.total_pages,
      loadProfileMoviePage
    );
    $("#profile-dialog").showModal();
  } catch (error) {
    toast(error.message);
  }
}

function renderMovieChoices(movies) {
  const container = $("#movie-choices");
  container.replaceChildren();
  movies.forEach((movie) => {
    const label = document.createElement("label");
    label.className = "movie-choice";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = movie.movie_id;
    checkbox.checked = state.favoriteMovies.has(movie.movie_id);
    checkbox.addEventListener("change", () => {
      if (checkbox.checked && state.favoriteMovies.size >= 20) {
        checkbox.checked = false;
        toast("Bạn có thể chọn tối đa 20 phim yêu thích.");
        return;
      }
      if (checkbox.checked) state.favoriteMovies.add(movie.movie_id);
      else state.favoriteMovies.delete(movie.movie_id);
    });
    const name = document.createElement("span");
    name.textContent = movie.title;
    label.append(checkbox, name);
    container.append(label);
  });
}

async function searchProfileMovies(event) {
  const query = event.currentTarget.value.trim();
  state.profileMovieQuery = query;
  state.profileMoviePage = 1;
  clearTimeout(state.profileSearchTimer);
  state.profileSearchTimer = setTimeout(loadProfileMoviePage, 250);
}

async function loadProfileMoviePage(page = state.profileMoviePage) {
  state.profileMoviePage = page;
  const data = await api(
    `/api/movies?q=${encodeURIComponent(state.profileMovieQuery)}&page=${page}&page_size=10`
  );
  renderMovieChoices(data.items);
  renderPagination(
    $("#profile-movie-pagination"),
    data.page,
    data.total_pages,
    loadProfileMoviePage
  );
}

async function saveProfile(event) {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  const payload = {
    display_name: form.get("display_name"),
    favorite_genres: form.getAll("favorite_genres"),
    favorite_movie_ids: [...state.favoriteMovies]
  };
  try {
    const data = await api("/api/profile", {
      method: "PUT",
      body: JSON.stringify(payload)
    });
    state.user = data.user;
    state.recommendationPage = 1;
    updateRoleUI();
    $("#profile-dialog").close();
    toast("Đã lưu sở thích của bạn.");
  } catch (error) {
    $("#profile-error").textContent = error.message;
  }
}

function addChatMessage(text, kind) {
  const bubble = document.createElement("div");
  bubble.className = `chat-bubble ${kind}-bubble`;
  bubble.textContent = text;
  $("#chat-messages").append(bubble);
  $("#chat-messages").scrollTop = $("#chat-messages").scrollHeight;
  return bubble;
}

async function sendChat(event) {
  event.preventDefault();
  const input = $("#chat-input");
  const message = input.value.trim();
  if (!message) return;
  addChatMessage(message, "user");
  input.value = "";
  const waiting = addChatMessage("Cine đang tìm phim phù hợp…", "bot");
  try {
    const data = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({ message })
    });
    waiting.textContent = data.reply;
  } catch (error) {
    waiting.textContent = error.message;
  }
}

async function loadAdminUsers() {
  try {
    const data = await api(`/api/admin/users?page=${state.adminUserPage}&page_size=10`);
    const users = data.items;
    const container = $("#admin-users");
    const sourceSummary = document.createElement("p");
    sourceSummary.className = "section-caption admin-source-summary";
    sourceSummary.textContent =
      `${data.dataset_user_count.toLocaleString("vi-VN")} user MovieLens trong tập ratings_small · ` +
      `${data.total.toLocaleString("vi-VN")} tài khoản ứng dụng`;
    const table = document.createElement("table");
    table.innerHTML = "<thead><tr><th>Người dùng</th><th>Vai trò</th><th>Trạng thái</th><th>Thao tác</th></tr></thead>";
    const body = document.createElement("tbody");
    users.forEach((user) => {
      const row = document.createElement("tr");
      const identity = document.createElement("td");
      identity.textContent = `${user.display_name} · ${user.email}`;
      const roleCell = document.createElement("td");
      const role = document.createElement("select");
      ["user", "admin"].forEach((value) => {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = value;
        option.selected = user.role === value;
        role.append(option);
      });
      roleCell.append(role);
      const statusCell = document.createElement("td");
      const disabled = document.createElement("input");
      disabled.type = "checkbox";
      disabled.checked = Boolean(user.disabled);
      disabled.setAttribute("aria-label", `Vô hiệu hóa ${user.email}`);
      statusCell.append(disabled, document.createTextNode(" Vô hiệu hóa"));
      const actions = document.createElement("td");
      const save = document.createElement("button");
      save.className = "button button-outline button-small";
      save.textContent = "Lưu";
      save.addEventListener("click", async () => {
        try {
          await api(`/api/admin/users/${user.id}`, {
            method: "PATCH",
            body: JSON.stringify({ role: role.value, disabled: disabled.checked })
          });
          toast("Đã cập nhật người dùng.");
        } catch (error) { toast(error.message); }
      });
      const remove = document.createElement("button");
      remove.className = "button button-danger button-small";
      remove.textContent = "Xóa";
      remove.addEventListener("click", async () => {
        if (!confirm(`Xóa tài khoản ${user.email}?`)) return;
        try {
          await api(`/api/admin/users/${user.id}`, { method: "DELETE" });
          loadAdminUsers();
          toast("Đã xóa tài khoản.");
        } catch (error) { toast(error.message); }
      });
      actions.append(save, remove);
      row.append(identity, roleCell, statusCell, actions);
      body.append(row);
    });
    table.append(body);
    container.replaceChildren(sourceSummary, table);
    renderPagination(
      $("#admin-users-pagination"),
      data.page,
      data.total_pages,
      (page) => {
        state.adminUserPage = page;
        loadAdminUsers();
      }
    );
  } catch (error) { toast(error.message); }
}

async function loadAdminMovies(query = state.adminMovieQuery, page = state.adminMoviePage) {
  try {
    state.adminMovieQuery = query;
    state.adminMoviePage = page;
    const data = await api(`/api/admin/movies?q=${encodeURIComponent(query)}&page=${page}&page_size=10`);
    const movies = data.items;
    const container = $("#admin-movies");
    const search = document.createElement("input");
    search.placeholder = "Tìm phim cần chỉnh sửa...";
    search.value = query;
    search.addEventListener("keydown", (event) => {
      if (event.key === "Enter") loadAdminMovies(search.value, 1);
    });
    const table = document.createElement("table");
    table.innerHTML = "<thead><tr><th>Phim</th><th>Thể loại (cách nhau bằng dấu phẩy)</th><th>Thao tác</th></tr></thead>";
    const body = document.createElement("tbody");
    movies.forEach((movie) => {
      const row = document.createElement("tr");
      const titleCell = document.createElement("td");
      const title = document.createElement("input");
      title.value = movie.title;
      const overview = document.createElement("textarea");
      overview.value = movie.overview;
      overview.placeholder = "Tóm tắt phim";
      titleCell.append(title, overview);
      const genreCell = document.createElement("td");
      const genres = document.createElement("input");
      genres.value = movie.genres.join(", ");
      genreCell.append(genres);
      const actions = document.createElement("td");
      const save = document.createElement("button");
      save.className = "button button-outline button-small";
      save.textContent = "Lưu";
      save.addEventListener("click", async () => {
        try {
          await api(`/api/admin/movies/${movie.movie_id}`, {
            method: "PATCH",
            body: JSON.stringify({
              title: title.value,
              genres: genres.value.split(",").map((item) => item.trim()).filter(Boolean),
              overview: overview.value
            })
          });
          toast("Đã cập nhật thông tin phim.");
        } catch (error) { toast(error.message); }
      });
      const remove = document.createElement("button");
      remove.className = "button button-danger button-small";
      remove.textContent = "Xóa";
      remove.addEventListener("click", async () => {
        if (!confirm(`Xóa phim ${movie.title} khỏi danh mục?`)) return;
        try {
          await api(`/api/admin/movies/${movie.movie_id}`, { method: "DELETE" });
          loadAdminMovies(search.value);
          toast("Đã xóa phim khỏi danh mục.");
        } catch (error) { toast(error.message); }
      });
      actions.append(save, remove);
      row.append(titleCell, genreCell, actions);
      body.append(row);
    });
    table.append(body);
    container.replaceChildren(search, table);
    renderPagination(
      $("#admin-movies-pagination"),
      data.page,
      data.total_pages,
      (nextPage) => loadAdminMovies(state.adminMovieQuery, nextPage)
    );
  } catch (error) { toast(error.message); }
}

function setupAdminTabs() {
  $$(".admin-tab").forEach((tab) => tab.addEventListener("click", () => {
    $$(".admin-tab").forEach((item) => item.classList.toggle("active", item === tab));
    const showUsers = tab.dataset.adminTab === "users";
    $("#admin-users").classList.toggle("hidden", !showUsers);
    $("#admin-users-pagination").classList.toggle("hidden", !showUsers);
    $("#admin-movies").classList.toggle("hidden", showUsers);
    $("#admin-movies-pagination").classList.toggle("hidden", showUsers);
    if (!showUsers) loadAdminMovies(state.adminMovieQuery, state.adminMoviePage);
  }));
}

function setup() {
  $("#login-open").addEventListener("click", () => { setAuthMode("login"); $("#auth-dialog").showModal(); });
  $("#register-open").addEventListener("click", () => { setAuthMode("register"); $("#auth-dialog").showModal(); });
  $("#auth-switch").addEventListener("click", () => setAuthMode(state.authMode === "login" ? "register" : "login"));
  $("#auth-form").addEventListener("submit", submitAuth);
  $("#logout-button").addEventListener("click", logout);
  $("#profile-open").addEventListener("click", openProfile);
  $("#profile-form").addEventListener("submit", saveProfile);
  $("#favorite-search").addEventListener("input", searchProfileMovies);
  $("#genre-filter").addEventListener("change", () => loadGuestContent(1));
  $("#chat-form").addEventListener("submit", sendChat);
  setupAdminTabs();
  Promise.all([loadGuestContent(), loadSession()]).catch((error) => toast(error.message));
}

setup();
